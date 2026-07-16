#![deny(unsafe_code)]
//! `kiln-proto`: types generated from `proto/kiln/v1/worker.proto` (the
//! gateway↔worker contract, SPEC §5) and `proto/kiln/v1/jobs.proto` (the
//! gateway↔job-runner control plane, SPEC §9.1).
//!
//! worker.proto wire semantics are frozen after Phase 2 — additive changes
//! only, via ADR. jobs.proto follows the same discipline once shipped.

/// Generated `kiln.v1` protocol types and gRPC service stubs.
pub mod v1 {
    tonic::include_proto!("kiln.v1");
}
