//! Gemini Bridge - AI/LLM Gemini Bridge & RAG Service
//!
//! A high-performance Rust service that integrates Google Gemini API
//! with Retrieval-Augmented Generation (RAG) capabilities.

pub mod config;
pub mod error;
pub mod handlers;
pub mod models;
pub mod services;

pub use config::{get_settings, Settings};
pub use error::{AppError, AppResult};
pub use handlers::{chat, health, rag};
pub use models::*;
pub use services::{GeminiService, RAGService, VectorStoreService};
