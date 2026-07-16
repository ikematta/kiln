fn main() {
    // rust-embed's derive requires the asset folder to exist at compile
    // time, but admin/build is npm output and gitignored — a cargo-only
    // checkout must still build (the UI then serves its "not built"
    // fallback). The rerun line makes release builds re-embed when the
    // SPA is rebuilt.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../admin/build");
    let _ = std::fs::create_dir_all(&dir);
    println!("cargo:rerun-if-changed=../../admin/build");
}
