//! Resumable Hugging Face downloads (SPEC §9.1: "resumable, `hf_hub` REST,
//! progress to stdout JSON lines").
//!
//! This is a port of the hardened downloader proven in
//! `scripts/fetch-test-model.sh` (CI run 29305594680: a silently dead socket
//! hung the old fetch for 1h42m), keeping its semantics bit-for-bit where
//! they matter:
//! - every request carries a per-read stall timeout ([`STALL_TIMEOUT`]) and a
//!   bounded retry budget ([`MAX_ATTEMPTS`], linear backoff) so a dead
//!   connection fails fast instead of blocking forever; a connection
//!   trickling bytes is bounded by the caller's outer timeout, not here;
//! - large files stream to a `.part` sibling and resume via HTTP Range;
//!   bytes are banked as they arrive, so the next attempt resumes from the
//!   true offset;
//! - a server that ignores Range (200 instead of 206) restarts the file; a
//!   416 discards the `.part` and refetches; a sha256 mismatch discards the
//!   `.part` — never retry on top of a corrupt prefix;
//! - files already present with the right size (and LFS sha256, when the
//!   tree lists one) are skipped;
//! - `HF_ENDPOINT` overrides the hub (mirrors; tests point it at a local
//!   stub server).
//!
//! Extensions over the script (it takes pinned commit shas; a general
//! download tool cannot): the requested revision is resolved to a commit sha
//! up front so interrupted-then-resumed downloads stay revision-coherent,
//! and tree listings follow `Link: rel="next"` pagination.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::events::{Event, Sink};

/// No bytes on the socket for this long = dead connection.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(30);
pub const MAX_ATTEMPTS: u32 = 4;
const RETRYABLE_HTTP: &[u16] = &[408, 425, 429, 500, 502, 503, 504];
/// Repo files never worth downloading (`.gitattributes`, README variants) —
/// same skip set the fetch script uses.
const SKIP_PREFIXES: &[&str] = &[".", "README"];
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
/// Written into the destination directory after a complete download.
pub const REVISION_MARKER: &str = ".kiln-revision";

#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("HTTP {status} for {url}")]
    Status { status: u16, url: String },
    #[error("giving up on {context} after {attempts} attempts: {last}")]
    Exhausted {
        context: String,
        attempts: u32,
        last: String,
    },
    #[error("hub API: {0}")]
    Api(String),
    #[error("http client: {0}")]
    Client(#[from] reqwest::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Per-attempt outcome classification, mirroring the script's exception
/// handling: transport errors, retryable statuses, short bodies, and sha
/// mismatches retry; other HTTP statuses are fatal.
enum AttemptError {
    Retry(String),
    Fatal(HubError),
}

impl From<HubError> for AttemptError {
    fn from(err: HubError) -> Self {
        AttemptError::Fatal(err)
    }
}

#[derive(Debug, Clone)]
pub struct TreeEntry {
    pub path: String,
    pub size: u64,
    /// sha256 of the content for LFS-tracked files; verified after download.
    pub lfs_sha256: Option<String>,
}

pub struct HubClient {
    http: reqwest::Client,
    endpoint: String,
    backoff_base: Duration,
}

impl HubClient {
    pub fn new(endpoint: impl Into<String>) -> Result<Self, HubError> {
        let http = reqwest::Client::builder()
            .connect_timeout(STALL_TIMEOUT)
            .read_timeout(STALL_TIMEOUT)
            .user_agent("kiln-jobs")
            .build()?;
        Ok(Self {
            http,
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            backoff_base: Duration::from_secs(5),
        })
    }

    /// Standard hub override knob (mirrors; the stub-server tests).
    pub fn from_env() -> Result<Self, HubError> {
        let endpoint =
            std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        Self::new(endpoint)
    }

    /// Test hook: shrink the linear backoff so retry paths run in
    /// milliseconds. Production keeps the script's 5s base.
    pub fn with_backoff_base(mut self, base: Duration) -> Self {
        self.backoff_base = base;
        self
    }

    async fn backoff(&self, attempt: u32) {
        if attempt > 1 {
            tokio::time::sleep(self.backoff_base * (attempt - 1)).await;
        }
    }

    /// GET returning the whole body, with the script's retry classification.
    async fn get_bytes_with_retry(
        &self,
        url: &str,
        context: &str,
        sink: &dyn Sink,
    ) -> Result<(Vec<u8>, Option<String>), HubError> {
        let mut last = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            self.backoff(attempt).await;
            match self.try_get_bytes(url).await {
                Ok(result) => return Ok(result),
                Err(AttemptError::Fatal(err)) => return Err(err),
                Err(AttemptError::Retry(err)) => {
                    sink.emit(&Event::Retry {
                        context: context.to_string(),
                        attempt,
                        max_attempts: MAX_ATTEMPTS,
                        error: err.clone(),
                    });
                    last = err;
                }
            }
        }
        Err(HubError::Exhausted {
            context: context.to_string(),
            attempts: MAX_ATTEMPTS,
            last,
        })
    }

    /// Body bytes plus the `Link: rel="next"` URL when the response is
    /// paginated.
    async fn try_get_bytes(&self, url: &str) -> Result<(Vec<u8>, Option<String>), AttemptError> {
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|err| AttemptError::Retry(err.to_string()))?;
        let status = response.status().as_u16();
        if RETRYABLE_HTTP.contains(&status) {
            return Err(AttemptError::Retry(format!("HTTP {status}")));
        }
        if !response.status().is_success() {
            return Err(AttemptError::Fatal(HubError::Status {
                status,
                url: url.to_string(),
            }));
        }
        let next = next_link(response.headers());
        let body = response
            .bytes()
            .await
            .map_err(|err| AttemptError::Retry(err.to_string()))?;
        Ok((body.to_vec(), next))
    }

    /// Resolves a branch/tag/commit ref to the commit sha it points at.
    pub async fn resolve_revision(
        &self,
        repo: &str,
        revision: &str,
        sink: &dyn Sink,
    ) -> Result<String, HubError> {
        let url = format!("{}/api/models/{repo}/revision/{revision}", self.endpoint);
        let (body, _) = self
            .get_bytes_with_retry(&url, &format!("revision {repo}@{revision}"), sink)
            .await?;
        let value: serde_json::Value = serde_json::from_slice(&body)
            .map_err(|err| HubError::Api(format!("bad revision response for {repo}: {err}")))?;
        value["sha"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| HubError::Api(format!("revision response for {repo} has no sha")))
    }

    /// Lists all files at a revision, following pagination.
    pub async fn list_tree(
        &self,
        repo: &str,
        revision: &str,
        sink: &dyn Sink,
    ) -> Result<Vec<TreeEntry>, HubError> {
        let mut url = format!(
            "{}/api/models/{repo}/tree/{revision}?recursive=true",
            self.endpoint
        );
        let mut entries = Vec::new();
        loop {
            let (body, next) = self
                .get_bytes_with_retry(&url, &format!("tree {repo}@{revision}"), sink)
                .await?;
            let listed: Vec<serde_json::Value> = serde_json::from_slice(&body)
                .map_err(|err| HubError::Api(format!("bad tree response for {repo}: {err}")))?;
            for item in listed {
                if item["type"].as_str() != Some("file") {
                    continue;
                }
                let Some(path) = item["path"].as_str() else {
                    continue;
                };
                let name = path.rsplit('/').next().unwrap_or(path);
                if SKIP_PREFIXES.iter().any(|p| name.starts_with(p)) {
                    continue;
                }
                entries.push(TreeEntry {
                    path: path.to_string(),
                    size: item["size"].as_u64().unwrap_or(0),
                    lfs_sha256: item["lfs"]["oid"].as_str().map(str::to_string),
                });
            }
            match next {
                Some(next_url) => url = next_url,
                None => return Ok(entries),
            }
        }
    }

    /// Downloads `repo@revision` into `dest`. Present-and-verified files are
    /// skipped; partial files resume. On success `dest/.kiln-revision`
    /// records `repo@sha`.
    pub async fn download_repo(
        &self,
        repo: &str,
        revision: &str,
        dest: &Path,
        sink: &dyn Sink,
    ) -> Result<(), HubError> {
        let sha = self.resolve_revision(repo, revision, sink).await?;
        let entries = self.list_tree(repo, &sha, sink).await?;
        if entries.is_empty() {
            return Err(HubError::Api(format!("no files listed for {repo}@{sha}")));
        }
        let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
        sink.emit(&Event::Plan {
            repo: repo.to_string(),
            revision: sha.clone(),
            files: entries.len(),
            total_bytes,
        });

        std::fs::create_dir_all(dest)?;
        let mut done_bytes: u64 = 0;
        for entry in &entries {
            let out = dest.join(&entry.path);
            if is_present_and_verified(&out, entry).await {
                sink.emit(&Event::Skip {
                    path: entry.path.clone(),
                });
                done_bytes += entry.size;
                continue;
            }
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let url = format!("{}/{repo}/resolve/{sha}/{}", self.endpoint, entry.path);
            self.download_file(&url, &out, entry, sink, done_bytes, total_bytes)
                .await?;
            done_bytes += entry.size;
        }

        std::fs::write(dest.join(REVISION_MARKER), format!("{repo}@{sha}\n"))?;
        sink.emit(&Event::Done {
            dest: dest.display().to_string(),
        });
        Ok(())
    }

    async fn download_file(
        &self,
        url: &str,
        out: &Path,
        entry: &TreeEntry,
        sink: &dyn Sink,
        done_before: u64,
        total_bytes: u64,
    ) -> Result<(), HubError> {
        let mut last = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            self.backoff(attempt).await;
            match self
                .try_download_file(url, out, entry, sink, done_before, total_bytes)
                .await
            {
                Ok(()) => return Ok(()),
                Err(AttemptError::Fatal(err)) => return Err(err),
                Err(AttemptError::Retry(err)) => {
                    sink.emit(&Event::Retry {
                        context: entry.path.clone(),
                        attempt,
                        max_attempts: MAX_ATTEMPTS,
                        error: err.clone(),
                    });
                    last = err;
                }
            }
        }
        Err(HubError::Exhausted {
            context: entry.path.clone(),
            attempts: MAX_ATTEMPTS,
            last,
        })
    }

    async fn try_download_file(
        &self,
        url: &str,
        out: &Path,
        entry: &TreeEntry,
        sink: &dyn Sink,
        done_before: u64,
        total_bytes: u64,
    ) -> Result<(), AttemptError> {
        let retry_io = |err: std::io::Error| AttemptError::Retry(err.to_string());
        let part = &part_path(out);

        // Touch: size-0 files skip the request but still get renamed into
        // place; everything else resumes from the banked length.
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(part)
            .map_err(retry_io)?;
        let mut offset = part.metadata().map_err(retry_io)?.len();
        if offset > entry.size {
            std::fs::remove_file(part).map_err(retry_io)?;
            offset = 0;
        }

        if offset < entry.size {
            sink.emit(&Event::File {
                path: entry.path.clone(),
                size: entry.size,
                resume_from: offset,
            });
            let mut request = self.http.get(url);
            if offset > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
            }
            let response = request
                .send()
                .await
                .map_err(|err| AttemptError::Retry(err.to_string()))?;
            let status = response.status();
            if status.as_u16() == 416 {
                // Resume window refused; start over.
                std::fs::remove_file(part).map_err(retry_io)?;
                return Err(AttemptError::Retry("HTTP 416: resume refused".into()));
            }
            if RETRYABLE_HTTP.contains(&status.as_u16()) {
                return Err(AttemptError::Retry(format!("HTTP {}", status.as_u16())));
            }
            if !status.is_success() {
                return Err(AttemptError::Fatal(HubError::Status {
                    status: status.as_u16(),
                    url: url.to_string(),
                }));
            }
            // 206 = server honored the Range; anything else ignored it, so
            // restart the file from zero.
            let append = offset > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
            if offset > 0 && !append {
                offset = 0;
            }
            let mut file = if append {
                std::fs::OpenOptions::new().append(true).open(part)
            } else {
                std::fs::File::create(part)
            }
            .map_err(retry_io)?;

            // Chunks are banked as they arrive (the script's read1
            // semantics): a stall mid-file loses nothing, and the next
            // attempt resumes from the true offset.
            let mut response = response;
            let mut received = offset;
            let mut last_progress = Instant::now();
            loop {
                let chunk = match response.chunk().await {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => break,
                    Err(err) => return Err(AttemptError::Retry(err.to_string())),
                };
                file.write_all(&chunk).map_err(retry_io)?;
                received += chunk.len() as u64;
                if last_progress.elapsed() >= Duration::from_secs(1) {
                    last_progress = Instant::now();
                    sink.emit(&Event::Progress {
                        path: entry.path.clone(),
                        received,
                        size: entry.size,
                        done_bytes: done_before + received,
                        total_bytes,
                    });
                }
            }
        }

        let got = part.metadata().map_err(retry_io)?.len();
        if got != entry.size {
            return Err(AttemptError::Retry(format!(
                "short body: got {got}, want {}",
                entry.size
            )));
        }
        if let Some(expected) = &entry.lfs_sha256
            && &sha256_file(part).await.map_err(retry_io)? != expected
        {
            // Never retry on top of a corrupt prefix.
            std::fs::remove_file(part).map_err(retry_io)?;
            return Err(AttemptError::Retry(
                "sha256 mismatch, partial discarded".into(),
            ));
        }
        std::fs::rename(part, out).map_err(retry_io)?;
        sink.emit(&Event::Progress {
            path: entry.path.clone(),
            received: entry.size,
            size: entry.size,
            done_bytes: done_before + entry.size,
            total_bytes,
        });
        Ok(())
    }
}

fn part_path(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    out.with_file_name(name)
}

/// Present with the expected size and (for LFS files) content hash.
async fn is_present_and_verified(out: &Path, entry: &TreeEntry) -> bool {
    let Ok(meta) = out.metadata() else {
        return false;
    };
    if meta.len() != entry.size {
        return false;
    }
    match &entry.lfs_sha256 {
        None => true,
        Some(expected) => matches!(sha256_file(out).await, Ok(actual) if &actual == expected),
    }
}

/// Streaming sha256 off the async threads (weight files are gigabytes).
async fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        use std::io::Read as _;
        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await
    .map_err(|err| std::io::Error::other(format!("sha256 task failed: {err}")))?
}

/// RFC 8288 `Link` header, `rel="next"` target (hub tree pagination).
fn next_link(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let value = headers.get(reqwest::header::LINK)?.to_str().ok()?;
    for part in value.split(',') {
        let mut sections = part.split(';');
        let target = sections.next()?.trim();
        let is_next = sections.any(|param| {
            param.trim().eq_ignore_ascii_case(r#"rel="next""#) || param.trim() == "rel=next"
        });
        if is_next {
            return Some(
                target
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .to_string(),
            );
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_path_appends_suffix() {
        assert_eq!(
            part_path(Path::new("/x/model.safetensors")),
            Path::new("/x/model.safetensors.part")
        );
    }

    #[test]
    fn link_header_next_target() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::LINK,
            r#"<https://hub/api/models/o/m/tree/r?cursor=abc>; rel="next""#
                .parse()
                .expect("header value"),
        );
        assert_eq!(
            next_link(&headers).as_deref(),
            Some("https://hub/api/models/o/m/tree/r?cursor=abc")
        );
        headers.clear();
        headers.insert(
            reqwest::header::LINK,
            r#"<https://hub/first>; rel="first", <https://hub/next>; rel="next""#
                .parse()
                .expect("header value"),
        );
        assert_eq!(next_link(&headers).as_deref(), Some("https://hub/next"));
        headers.clear();
        assert_eq!(next_link(&headers), None);
    }
}
