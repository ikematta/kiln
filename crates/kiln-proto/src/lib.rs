#![deny(unsafe_code)]
//! `kiln-proto`: types generated from `proto/kiln/v1/worker.proto`.
//!
//! This is the single contract between the gateway and any worker (SPEC §5).
//! Wire semantics are frozen after Phase 2 — additive changes only, via ADR.

/// Generated `kiln.v1` protocol types and gRPC service stubs.
pub mod v1 {
    tonic::include_proto!("kiln.v1");
}
