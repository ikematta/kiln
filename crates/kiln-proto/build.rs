fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../proto/kiln/v1/worker.proto");
    println!("cargo:rerun-if-changed=../../proto/kiln/v1/jobs.proto");
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "../../proto/kiln/v1/worker.proto",
                "../../proto/kiln/v1/jobs.proto",
            ],
            &["../../proto"],
        )?;
    Ok(())
}
