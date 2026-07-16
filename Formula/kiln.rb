# Kiln — Homebrew formula (SPEC §12 Phase 10; §1.1 goal 8: "installable via
# Homebrew; runnable as a launchd service").
#
# Build-from-source only for now (no bottles, no tagged release yet):
#   brew install --build-from-source ./Formula/kiln.rb
#
# Layout it installs:
#   bin/            kiln, kiln-gateway, kiln-worker, kiln-jobs
#   etc/kiln/       kiln.toml (live config; preserved across reinstalls),
#                   kiln.toml.example (annotated reference)
#   libexec/        kiln_worker_py/ + kiln_jobs_py/ (uv-synced venvs from the
#                   repo's own uv.lock pins), scripts/bench.{sh,py}
#   var/log/        kiln.log (when run via `brew services`)
#
# The launchd integration is the `service do` block: `brew services start
# kiln` generates and loads the plist. packaging/dev.kiln.gateway.plist is
# the equivalent template for non-Homebrew installs.
class Kiln < Formula
  desc "LLM inference server for Apple Silicon: Rust control plane over MLX"
  homepage "https://github.com/ikematta/kiln"
  # No tagged release yet: stable tracks main (version = workspace version).
  # When a release is tagged, pin `tag:`/`revision:` here instead.
  url "https://github.com/ikematta/kiln.git", branch: "main", using: :git
  version "0.0.1"
  head "https://github.com/ikematta/kiln.git", branch: "main"

  depends_on "cmake" => :build # vendored mlx-c submodule
  depends_on "node@22" => :build # admin UI (embedded into the release gateway)
  depends_on "protobuf" => :build # protoc, for tonic-prost-build codegen
  # rustup, NOT brew's floating rust: the repo pins its exact toolchain in
  # rust-toolchain.toml over a known rustc miscompilation
  # (rust-lang/rust#158830); rustup honors that pin, a floating rust would
  # silently bypass it.
  depends_on "rustup" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.12" # worker/jobs venv interpreter (opt path is stable)
  depends_on "uv" # venv sync at install; `kiln-jobs quantize` invokes `uv run`

  def install
    # Keep every toolchain cache inside the build sandbox.
    ENV["RUSTUP_HOME"] = buildpath/".rustup"
    ENV["CARGO_HOME"] = buildpath/".cargo"
    ENV["UV_CACHE_DIR"] = buildpath/".uv-cache"
    ENV["UV_PYTHON"] = formula_opt_bin("python@3.12")/"python3.12"
    ENV["npm_config_cache"] = buildpath/".npm"
    ENV.prepend_path "PATH", formula_opt_bin("rustup")
    ENV.prepend_path "PATH", buildpath/".cargo/bin"

    # Installs the exact toolchain named by rust-toolchain.toml.
    system "rustup", "toolchain", "install"

    # The admin SPA must exist before the release build: rust-embed embeds
    # admin/build into the gateway binary at compile time (SPEC §1.1
    # "single static gateway binary").
    system "npm", "ci", "--prefix", "admin"
    system "npm", "run", "build", "--prefix", "admin"

    # Requires a Metal-capable compiler toolchain for the mlx-c kernels
    # (Xcode, or Command Line Tools with the Metal toolchain component).
    # A shared target dir means one real build; the later installs reuse
    # its artifacts. --locked (std_cargo_args) enforces Cargo.lock.
    ENV["CARGO_TARGET_DIR"] = buildpath/"target"
    %w[kiln-cli kiln-gateway kiln-worker kiln-jobs].each do |crate|
      system "cargo", "install", *std_cargo_args(path: "crates/#{crate}")
    end

    # Python environments, synced from the repo's committed uv.lock so the
    # installed pins are byte-identical to the tested ones (mlx/mlx-lm pins
    # move only in lockstep with the vendored mlx-c — PROGRESS 2026-07-03).
    libexec.install "python/kiln_worker_py", "python/kiln_jobs_py"
    system "uv", "sync", "--project", libexec/"kiln_worker_py", "--frozen", "--no-dev"
    system "uv", "sync", "--project", libexec/"kiln_jobs_py", "--frozen", "--no-dev"

    (libexec/"scripts").install "scripts/bench.sh", "scripts/bench.py"

    # Live config (preserved across reinstalls) + the annotated reference.
    (buildpath/"kiln.toml.default").write default_kiln_toml
    (etc/"kiln").install "kiln.toml.default" => "kiln.toml"
    (etc/"kiln").install "kiln.toml.example"

    doc.install "README.md", "docs/CONFIGURATION.md", "docs/API_COMPAT.md"
  end

  def default_kiln_toml
    <<~TOML
      # Kiln configuration (Homebrew install).
      # Field reference: #{opt_doc}/CONFIGURATION.md
      # Annotated example: #{etc}/kiln/kiln.toml.example

      [server]
      host = "127.0.0.1"
      port = 8080
      # Installed layout: the Python fallback worker and the jobs venv live
      # under libexec (the checkout defaults assume `uv run` from a repo).
      python_worker_argv = ["#{opt_libexec}/kiln_worker_py/.venv/bin/python", "-m", "kiln_worker_py"]
      jobs_argv = ["#{opt_bin}/kiln-jobs", "--venv", "#{opt_libexec}/kiln_jobs_py"]

      # One [[model]] block per served model. `path` is a Hugging Face repo
      # id (downloaded on first load) or a local directory, e.g.:
      #
      # [[model]]
      # id = "llama-3.2-1b"
      # path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
      # worker = "auto"

      [auth]
      # The admin API (and the /ui dashboard's data) stays disabled (403)
      # until this is set. Hash a token with: kiln-gateway hash-key
      admin_token_hash = ""
    TOML
  end

  service do
    run [opt_bin/"kiln", "serve", "--config", etc/"kiln/kiln.toml"]
    keep_alive true
    working_dir var
    log_path var/"log/kiln.log"
    error_log_path var/"log/kiln.log"
  end

  def caveats
    <<~EOS
      Configure models in #{etc}/kiln/kiln.toml, then run in the foreground:
        kiln serve
      or as a launchd service:
        brew services start kiln

      Download a model:  kiln-jobs download mlx-community/Llama-3.2-1B-Instruct-4bit
      Admin API/UI:      set auth.admin_token_hash (hash a token with
                         `kiln-gateway hash-key`), then see http://127.0.0.1:8080/ui
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/kiln --version")
    # hash-key exercises the gateway binary end to end (argon2 PHC output).
    assert_match "$argon2", shell_output("#{bin}/kiln-gateway hash-key test-token")
    assert_match "usage", shell_output("#{bin}/kiln-jobs 2>&1", 2)
  end
end
