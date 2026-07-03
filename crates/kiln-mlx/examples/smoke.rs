//! Phase 0 acceptance: `cargo run -p kiln-mlx --example smoke` must print `3.0`.

#[cfg(feature = "metal")]
fn main() {
    let value = kiln_mlx::smoke::add_scalars(1.0, 2.0).expect("mlx smoke test failed");
    println!("{value:?}");
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("kiln-mlx was built without the `metal` feature; nothing to smoke-test.");
    std::process::exit(1);
}
