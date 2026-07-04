//! Builds the vendored mlx-c (which FetchContent-pins and builds MLX itself)
//! via cmake, links the resulting static archives, and bindgen-generates the
//! raw FFI surface from the pinned headers (SPEC §7.1) into
//! `$OUT_DIR/mlx_sys.rs`, included by `src/sys.rs`.
//!
//! Skipped entirely when the `metal` feature is off — that is the Linux CI
//! compile-check path (`cargo build --workspace --no-default-features`).
//!
//! The cmake tree lives at `target/mlx-c-build` (documented in CLAUDE.md); a
//! stale-build recovery is `rm -rf target/mlx-c-build && cargo build -p kiln-mlx`.

use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/mlx-c/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=KILN_MLX_METAL");

    if env::var_os("CARGO_FEATURE_METAL").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let vendor = manifest_dir.join("vendor/mlx-c");
    if !vendor.join("CMakeLists.txt").exists() {
        panic!(
            "vendored mlx-c not found at {}; run `git submodule update --init --recursive`",
            vendor.display()
        );
    }

    let build_root = match env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => manifest_dir.join("../../target"),
    }
    .join("mlx-c-build");

    generate_bindings(&vendor);

    let metal = metal_enabled();
    println!(
        "cargo:warning=kiln-mlx: building vendored mlx-c (MLX_BUILD_METAL={})",
        if metal { "ON" } else { "OFF" }
    );

    let dest = cmake::Config::new(&vendor)
        .out_dir(&build_root)
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_METAL", if metal { "ON" } else { "OFF" })
        // Subprojects fetched by MLX may declare cmake_minimum_required < 3.5,
        // which CMake 4.x refuses without this escape hatch.
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        .build_target("mlxc")
        .build();

    // Link every static archive the build tree produced (libmlxc.a, libmlx.a,
    // plus any static deps MLX builds), then the system frameworks MLX needs.
    let build_tree = dest.join("build");
    let mut search_dirs = BTreeSet::new();
    let mut libs = BTreeSet::new();
    collect_archives(&build_tree, &mut search_dirs, &mut libs);
    for required in ["mlxc", "mlx"] {
        assert!(
            libs.contains(required),
            "mlx-c build did not produce lib{required}.a under {}",
            build_tree.display()
        );
    }

    for dir in &search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    for lib in &libs {
        if lib != "mlxc" && lib != "mlx" {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    if metal {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }
}

/// Bindgen the entire `mlx_*` C surface from the vendored headers at the pin.
/// The headers are frozen with the submodule (ADR 0001), so the generated
/// bindings are reproducible; regenerating on every build keeps them honest.
fn generate_bindings(vendor: &Path) {
    let header = vendor.join("mlx/c/mlx.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let bindings = bindgen::Builder::default()
        .header(header.display().to_string())
        .clang_arg(format!("-I{}", vendor.display()))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .layout_tests(false)
        .rust_edition(bindgen::RustEdition::Edition2024)
        .generate()
        .expect("bindgen failed on vendored mlx-c headers");
    bindings
        .write_to_file(out.join("mlx_sys.rs"))
        .expect("failed to write mlx_sys.rs");
}

/// Metal kernel compilation needs the Metal toolchain (full Xcode); plain
/// Command Line Tools machines must build MLX CPU-only. `KILN_MLX_METAL=0|1`
/// overrides autodetection.
fn metal_enabled() -> bool {
    match env::var("KILN_MLX_METAL").as_deref() {
        Ok("0") => return false,
        Ok("1") => return true,
        _ => {}
    }
    Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .is_ok_and(|out| out.status.success())
}

fn collect_archives(dir: &Path, search_dirs: &mut BTreeSet<PathBuf>, libs: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_archives(&path, search_dirs, libs);
        } else if let (Some(stem), Some("a")) = (
            path.file_stem().and_then(|s| s.to_str()),
            path.extension().and_then(|e| e.to_str()),
        ) && let Some(name) = stem.strip_prefix("lib")
        {
            libs.insert(name.to_string());
            if let Some(parent) = path.parent() {
                search_dirs.insert(parent.to_path_buf());
            }
        }
    }
}
