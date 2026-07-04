#![deny(unsafe_code)]
//! `kiln-gateway` library: configuration, model registry, worker supervision,
//! and the axum HTTP surface (SPEC §8).

pub mod config;
pub mod metrics;
pub mod registry;
pub mod supervisor;
pub mod uds;
