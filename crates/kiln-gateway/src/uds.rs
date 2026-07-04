//! Lazy tonic channels over Unix domain sockets (SPEC §3: gRPC/UDS only;
//! workers are never network-exposed).

use std::path::PathBuf;
use std::time::Duration;

use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use tonic::transport::{Channel, Endpoint, Uri};

/// Builds a channel that (re)connects to `socket_path` on demand. Connections
/// are established lazily per RPC attempt, so the same channel keeps working
/// across worker restarts on the same socket path.
pub fn uds_channel(socket_path: PathBuf) -> Result<Channel, tonic::transport::Error> {
    // The URI is required by the HTTP/2 layer but never resolved; the
    // connector below ignores it and dials the socket path instead.
    let endpoint =
        Endpoint::try_from("http://kiln-worker.invalid")?.connect_timeout(Duration::from_secs(2));
    Ok(
        endpoint.connect_with_connector_lazy(tower::service_fn(move |_: Uri| {
            let path = socket_path.clone();
            async move { Ok::<_, std::io::Error>(TokioIo::new(UnixStream::connect(path).await?)) }
        })),
    )
}
