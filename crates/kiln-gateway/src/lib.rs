#![deny(unsafe_code)]
//! `kiln-gateway` library: configuration, model registry, worker supervision,
//! and the axum HTTP surface (SPEC §8).

pub mod admin;
pub mod admin_models;
pub mod admin_register;
pub mod anthropic;
pub mod app;
pub mod auth;
pub mod chat;
pub mod completions;
pub mod config;
pub mod config_write;
pub mod error;
pub mod lifecycle;
pub mod messages;
pub mod metrics;
pub mod openai;
pub mod registry;
pub mod supervisor;
pub mod sysmem;
pub mod uds;
pub mod ui;
