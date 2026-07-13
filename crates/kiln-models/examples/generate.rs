//! Single-stream generation benchmark/demo for the Phase 3 acceptance
//! numbers. Run in release mode against a local model directory:
//!
//! ```sh
//! cargo run --release -p kiln-models --example generate -- \
//!     --model ~/.kiln/test-models/llama-3.2-1b-4bit \
//!     --prompt "Write a story about Einstein" --max-tokens 256 --chat
//! ```

#[cfg(feature = "metal")]
fn main() {
    use kiln_engine::Sampler;
    use kiln_mlx::Stream;
    use kiln_models::{LlamaModel, generate};
    use kiln_tokenize::{ChatMessage, ChatTemplate, Tokenizer};

    let mut model_dir = None;
    let mut prompt = "Write a story about Einstein".to_owned();
    let mut max_tokens = 256_usize;
    let mut chat = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = Some(args.next().expect("--model needs a path")),
            "--prompt" => prompt = args.next().expect("--prompt needs text"),
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .expect("--max-tokens needs a number")
                    .parse()
                    .expect("--max-tokens must be a positive integer");
            }
            "--chat" => chat = true,
            other => panic!("unknown argument {other:?}"),
        }
    }
    let model_dir = model_dir.expect("--model <dir> is required");

    let load_start = std::time::Instant::now();
    let stream = Stream::gpu();
    let model = LlamaModel::load(&model_dir, &stream).expect("model loads");
    let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("tokenizer loads");
    eprintln!(
        "loaded {model_dir} in {:.2}s",
        load_start.elapsed().as_secs_f64()
    );

    let prompt_ids = if chat {
        let template = ChatTemplate::from_model_dir(&model_dir).expect("template loads");
        let rendered = template
            .render(&[ChatMessage::text("user", prompt)], true)
            .expect("renders");
        tokenizer.encode(&rendered, false).expect("encodes")
    } else {
        tokenizer.encode(&prompt, true).expect("encodes")
    };

    let mut sampler = Sampler::greedy().expect("sampler");
    let output = generate(
        &model,
        &prompt_ids,
        max_tokens,
        |logprobs, s| sampler.sample(logprobs, s),
        &stream,
    )
    .expect("generates");

    let text = tokenizer.decode(&output.tokens, false).expect("decodes");
    println!("{text}");
    eprintln!("==========");
    eprintln!(
        "Prompt: {} tokens, first token in {:.3}s ({:.1} tok/s prefill+first)",
        prompt_ids.len(),
        output.prefill_seconds,
        prompt_ids.len() as f64 / output.prefill_seconds,
    );
    eprintln!(
        "Generation: {} tokens, {:.1} tok/s decode",
        output.tokens.len(),
        output.decode_tokens_per_sec(),
    );
    eprintln!(
        "Peak MLX memory: {:.3} GB",
        kiln_mlx::memory::peak_memory().unwrap_or(0) as f64 / 1e9
    );
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("kiln-models was built without the `metal` feature.");
    std::process::exit(1);
}
