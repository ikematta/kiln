#![deny(unsafe_code)]
//! `kiln-gateway` library: configuration, model registry, worker supervision,
//! and the axum HTTP surface (SPEC §8).

pub mod admin;
pub mod anthropic;
pub mod app;
pub mod auth;
pub mod chat;
pub mod completions;
pub mod config;
pub mod error;
pub mod lifecycle;
pub mod messages;
pub mod metrics;
pub mod openai;
pub mod registry;
pub mod supervisor;
pub mod uds;
