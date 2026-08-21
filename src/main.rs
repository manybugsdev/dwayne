use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as base64_engine;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

/// Repeatedly ask an AI to "replicate" an image, feeding each result back in
/// as the next input, to watch it drift over many generations.
///
/// Example:
///   dwayne input.png -o output.png --epoch=101 \
///     --prompt="Create the exact replica of this image, don't change a thing"
#[derive(Parser, Debug)]
#[command(name = "dwayne", version, about)]
struct Cli {
    /// Input image (png/jpg/webp)
    input: PathBuf,

    /// Where to write the final image
    #[arg(short = 'o', long, default_value = "output.png")]
    output: PathBuf,

    /// Number of times to feed the image back through the model
    #[arg(long, default_value_t = 1)]
    epoch: u32,

    /// Instruction sent to the model on every generation
    #[arg(
        long,
        default_value = "Create the exact replica of this image, don't change a thing"
    )]
    prompt: String,

    /// OpenAI Responses API model to use (must support the image_generation tool, e.g. gpt-5, gpt-4o)
    #[arg(long, default_value = "gpt-5")]
    model: String,

    /// Output image size passed to the API (e.g. 1024x1024, 1024x1536, 1536x1024, auto)
    #[arg(long, default_value = "auto")]
    size: String,

    /// OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    api_key: String,

    /// Directory to also save every intermediate generation into (epoch_000.png, epoch_001.png, ...)
    #[arg(long)]
    save_intermediates: Option<PathBuf>,
}

#[derive(Deserialize)]
struct ResponsesEnvelope {
    output: Vec<OutputItem>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    ImageGenerationCall {
        status: String,
        result: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct ApiErrorEnvelope {
    error: ApiError,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
}

fn guess_mime(bytes: &[u8], path: &PathBuf) -> &'static str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return "image/png";
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "image/jpeg";
    }
    if bytes.len() >= 12 && &bytes[8..12] == b"WEBP" {
        return "image/webp";
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
    {
        Some(ext) if ext == "jpg" || ext == "jpeg" => "image/jpeg",
        Some(ext) if ext == "webp" => "image/webp",
        _ => "image/png",
    }
}

async fn replicate_image(
    client: &reqwest::Client,
    cli: &Cli,
    image_bytes: Vec<u8>,
    mime: &'static str,
) -> Result<Vec<u8>> {
    const MAX_ATTEMPTS: u32 = 5;
    let mut attempt = 0;

    loop {
        attempt += 1;

        let image_url = format!(
            "data:{mime};base64,{}",
            base64_engine.encode(&image_bytes)
        );

        let body = serde_json::json!({
            "model": cli.model,
            "input": [{
                "role": "user",
                "content": [
                    { "type": "input_text", "text": cli.prompt },
                    { "type": "input_image", "image_url": image_url },
                ],
            }],
            "tools": [{
                "type": "image_generation",
                "size": cli.size,
                "quality": "auto",
            }],
        });

        let response = client
            .post("https://api.openai.com/v1/responses")
            .bearer_auth(&cli.api_key)
            .json(&body)
            .send()
            .await
            .context("failed to reach OpenAI API")?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .context("failed to read OpenAI API response body")?;

        if status.is_success() {
            let parsed: ResponsesEnvelope = serde_json::from_slice(&body)
                .context("failed to parse OpenAI API response")?;
            let call = parsed
                .output
                .into_iter()
                .find_map(|item| match item {
                    OutputItem::ImageGenerationCall { status, result } => {
                        Some((status, result))
                    }
                    OutputItem::Other => None,
                })
                .ok_or_else(|| anyhow!("OpenAI API response contained no image_generation_call"))?;
            let (call_status, result) = call;
            if call_status != "completed" {
                bail!("image generation did not complete (status: {call_status})");
            }
            let b64 = result
                .ok_or_else(|| anyhow!("OpenAI API response contained no image data"))?;
            return base64_engine
                .decode(b64)
                .context("failed to decode base64 image data");
        }

        let message = serde_json::from_slice::<ApiErrorEnvelope>(&body)
            .map(|e| e.error.message)
            .unwrap_or_else(|_| String::from_utf8_lossy(&body).to_string());

        let retryable = status.as_u16() == 429 || status.is_server_error();
        if retryable && attempt < MAX_ATTEMPTS {
            let backoff = Duration::from_secs(2u64.pow(attempt));
            eprintln!(
                "  request failed ({status}): {message} — retrying in {}s ({attempt}/{MAX_ATTEMPTS})",
                backoff.as_secs()
            );
            tokio::time::sleep(backoff).await;
            continue;
        }

        bail!("OpenAI API request failed ({status}): {message}");
    }
}

async fn run(cli: Cli) -> Result<()> {
    if cli.epoch == 0 {
        bail!("--epoch must be at least 1");
    }

    let mut image_bytes = std::fs::read(&cli.input)
        .with_context(|| format!("failed to read input image {}", cli.input.display()))?;
    let mut mime = guess_mime(&image_bytes, &cli.input);

    if let Some(dir) = &cli.save_intermediates {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create directory {}", dir.display()))?;
        std::fs::write(dir.join("epoch_000.png"), &image_bytes)
            .context("failed to write intermediate image")?;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .context("failed to build HTTP client")?;

    let bar = ProgressBar::new(cli.epoch as u64);
    bar.set_style(
        ProgressStyle::with_template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );
    bar.set_message("replicating");

    for epoch in 1..=cli.epoch {
        image_bytes = replicate_image(&client, &cli, image_bytes, mime)
            .await
            .with_context(|| format!("generation {epoch}/{} failed", cli.epoch))?;
        // Every generation comes back from the API as PNG.
        mime = "image/png";

        if let Some(dir) = &cli.save_intermediates {
            let path = dir.join(format!("epoch_{epoch:03}.png"));
            std::fs::write(&path, &image_bytes)
                .with_context(|| format!("failed to write {}", path.display()))?;
        }

        bar.inc(1);
    }
    bar.finish_with_message("done");

    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }
    std::fs::write(&cli.output, &image_bytes)
        .with_context(|| format!("failed to write output image {}", cli.output.display()))?;

    println!("wrote {}", cli.output.display());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli).await
}
