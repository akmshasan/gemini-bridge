use crate::config::Settings;
use crate::error::{AppError, AppResult};
use crate::models::DocumentChunkResult;
use reqwest::{Client, Method, RequestBuilder, StatusCode};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

const PREVIEW_LENGTH: usize = 120;
const DEFAULT_HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(5);
const SCROLL_PAGE_LIMIT: usize = 1000;

/// Talks to a real Qdrant instance over its REST API (the same API surface
/// exposed on the port `qdrant_url` points at, e.g. `http://localhost:6333`).
pub struct VectorStoreService {
    settings: Settings,
    http_client: Client,
}

impl VectorStoreService {
    pub async fn new(settings: Settings) -> AppResult<Self> {
        Self::new_with_timeout(settings, DEFAULT_HEALTH_CHECK_TIMEOUT).await
    }

    /// Same as `new`, but with a configurable health-check timeout so tests
    /// can exercise the timeout branch without waiting on the real default.
    async fn new_with_timeout(settings: Settings, timeout: Duration) -> AppResult<Self> {
        let http_client = Client::new();
        let service = Self {
            settings,
            http_client,
        };

        // Lenient liveness probe: Qdrant might still be starting up, so a
        // failure here only gets logged - it doesn't fail construction. Real
        // errors surface properly on the first actual read/write below.
        let health_url = format!("{}/healthz", service.settings.qdrant_url);
        match tokio::time::timeout(timeout, service.authed(Method::GET, &health_url).send()).await
        {
            Ok(Ok(_response)) => {
                tracing::info!("Qdrant connection verified");
            }
            Ok(Err(e)) => {
                tracing::warn!("Qdrant health check error (will retry on first use): {}", e);
            }
            Err(_timeout) => {
                tracing::warn!("Qdrant health check timed out (will retry on first use)");
            }
        }

        Ok(service)
    }

    fn collection_url(&self) -> String {
        format!(
            "{}/collections/{}",
            self.settings.qdrant_url, self.settings.collection_name
        )
    }

    /// Builds a request with the `api-key` header attached when one is
    /// configured, per Qdrant's authentication convention.
    fn authed(&self, method: Method, url: &str) -> RequestBuilder {
        let mut builder = self.http_client.request(method, url);
        if let Some(key) = &self.settings.qdrant_api_key {
            builder = builder.header("api-key", key);
        }
        builder
    }

    /// Split text into overlapping chunks for indexing.
    pub fn chunk_text(
        &self,
        text: &str,
        chunk_size: Option<usize>,
        chunk_overlap: Option<usize>,
    ) -> Vec<String> {
        let size = chunk_size.unwrap_or(self.settings.default_chunk_size);
        let overlap = chunk_overlap.unwrap_or(self.settings.default_chunk_overlap);

        let text = text.trim();
        if text.is_empty() {
            return Vec::new();
        }

        if text.len() <= size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + size).min(text.len());
            let chunk = &text[start..end];
            let trimmed = chunk.trim().to_string();
            if !trimmed.is_empty() {
                chunks.push(trimmed);
            }
            start += size.saturating_sub(overlap);
        }

        chunks
    }

    async fn collection_exists(&self) -> AppResult<bool> {
        let resp = self
            .authed(Method::GET, &self.collection_url())
            .send()
            .await
            .map_err(|e| AppError::QdrantError(format!("Failed to reach Qdrant: {}", e)))?;
        Ok(resp.status().is_success())
    }

    async fn ensure_collection(&self, vector_size: usize) -> AppResult<()> {
        if self.collection_exists().await? {
            return Ok(());
        }

        let body = json!({
            "vectors": { "size": vector_size, "distance": "Cosine" }
        });

        let resp = self
            .authed(Method::PUT, &self.collection_url())
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::QdrantError(format!("Failed to create collection: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AppError::QdrantError(format!(
                "Failed to create collection ({}): {}",
                status, text
            )));
        }

        Ok(())
    }

    /// Add text chunks with their precomputed embeddings into Qdrant.
    pub async fn add_document(
        &self,
        chunks: &[String],
        source: &str,
        embeddings: Vec<Vec<f32>>,
        custom_metadata: Option<HashMap<String, String>>,
    ) -> AppResult<Vec<String>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let vector_size = embeddings.first().map(|e| e.len()).unwrap_or(0);
        if vector_size == 0 {
            return Err(AppError::ValidationError(
                "Cannot index chunks without embeddings".to_string(),
            ));
        }

        self.ensure_collection(vector_size).await?;

        let metadata = custom_metadata.unwrap_or_default();
        let mut chunk_ids = Vec::with_capacity(chunks.len());
        let mut points = Vec::with_capacity(chunks.len());

        for (idx, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
            let chunk_id = Uuid::new_v4().to_string();
            chunk_ids.push(chunk_id.clone());

            points.push(json!({
                "id": chunk_id,
                "vector": embedding,
                "payload": {
                    "source": source,
                    "content": chunk,
                    "chunk_index": idx,
                    "metadata": metadata,
                }
            }));
        }

        let url = format!("{}/points?wait=true", self.collection_url());
        let resp = self
            .authed(Method::PUT, &url)
            .json(&json!({ "points": points }))
            .send()
            .await
            .map_err(|e| AppError::QdrantError(format!("Failed to upsert points: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AppError::QdrantError(format!(
                "Failed to upsert points ({}): {}",
                status, text
            )));
        }

        tracing::debug!(
            "Added {} chunks from source '{}' to Qdrant collection '{}'",
            chunks.len(),
            source,
            self.settings.collection_name
        );

        Ok(chunk_ids)
    }

    /// Retrieve the top-k most similar document chunks.
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> AppResult<Vec<DocumentChunkResult>> {
        if !self.collection_exists().await? {
            // Nothing has ever been ingested - an empty result is correct,
            // not an error.
            return Ok(Vec::new());
        }

        let url = format!("{}/points/search", self.collection_url());
        let body = json!({
            "vector": query_embedding,
            "limit": top_k,
            "with_payload": true,
        });

        let resp = self
            .authed(Method::POST, &url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::QdrantError(format!("Search request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AppError::QdrantError(format!(
                "Search failed ({}): {}",
                status, text
            )));
        }

        let body: Value = resp
            .json()
            .await
            .map_err(|e| AppError::QdrantError(format!("Failed to parse search response: {}", e)))?;

        let results = body["result"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(Self::point_to_chunk_result)
            .collect();

        tracing::debug!("Searched Qdrant for {} similar chunks", top_k);

        Ok(results)
    }

    /// List all indexed document chunks in the vector store.
    pub async fn list_documents(&self) -> AppResult<Vec<(String, String, usize, String)>> {
        if !self.collection_exists().await? {
            return Ok(Vec::new());
        }

        let url = format!("{}/points/scroll", self.collection_url());
        let body = json!({
            "limit": SCROLL_PAGE_LIMIT,
            "with_payload": true,
            "with_vector": false,
        });

        let resp = self
            .authed(Method::POST, &url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::QdrantError(format!("Scroll request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AppError::QdrantError(format!(
                "Scroll failed ({}): {}",
                status, text
            )));
        }

        let body: Value = resp
            .json()
            .await
            .map_err(|e| AppError::QdrantError(format!("Failed to parse scroll response: {}", e)))?;

        let points = body["result"]["points"].as_array().cloned().unwrap_or_default();

        let documents = points
            .iter()
            .map(|point| {
                let chunk_id = Self::point_id(point);
                let source = point["payload"]["source"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let content = point["payload"]["content"].as_str().unwrap_or_default();
                let preview: String = content.chars().take(PREVIEW_LENGTH).collect();
                (chunk_id, source, content.len(), preview)
            })
            .collect();

        tracing::debug!("Listed indexed documents from Qdrant");

        Ok(documents)
    }

    /// Clear all indexed documents from the collection.
    pub async fn clear(&self) -> AppResult<()> {
        let resp = self
            .authed(Method::DELETE, &self.collection_url())
            .send()
            .await
            .map_err(|e| AppError::QdrantError(format!("Failed to delete collection: {}", e)))?;

        // A collection that never existed is already "cleared".
        if resp.status().is_success() || resp.status() == StatusCode::NOT_FOUND {
            tracing::debug!(
                "Cleared Qdrant collection '{}'",
                self.settings.collection_name
            );
            return Ok(());
        }

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Err(AppError::QdrantError(format!(
            "Failed to clear collection ({}): {}",
            status, text
        )))
    }

    fn point_id(point: &Value) -> String {
        match &point["id"] {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    fn point_to_chunk_result(point: &Value) -> DocumentChunkResult {
        let chunk_id = Self::point_id(point);
        let payload = &point["payload"];
        let source = payload["source"].as_str().unwrap_or_default().to_string();
        let content = payload["content"].as_str().unwrap_or_default().to_string();
        let metadata = payload["metadata"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let distance = point["score"].as_f64().map(|s| s as f32);

        DocumentChunkResult {
            chunk_id,
            content,
            source,
            distance,
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service_with(settings: Settings) -> VectorStoreService {
        VectorStoreService {
            settings,
            http_client: Client::new(),
        }
    }

    fn settings_pointing_at(qdrant_url: &str) -> Settings {
        let mut s = Settings::default();
        s.qdrant_url = qdrant_url.to_string();
        s
    }

    // ---- chunk_text (pure logic, unaffected by the Qdrant backend) ----

    #[test]
    fn test_chunk_text_empty_input() {
        let service = service_with(Settings::default());
        assert!(service.chunk_text("   ", None, None).is_empty());
        assert!(service.chunk_text("", None, None).is_empty());
    }

    #[test]
    fn test_chunk_text_shorter_than_chunk_size() {
        let service = service_with(Settings::default());
        let chunks = service.chunk_text("Hello world", None, None);
        assert_eq!(chunks, vec!["Hello world".to_string()]);
    }

    #[test]
    fn test_chunk_text_exact_boundary() {
        let service = service_with(Settings::default());
        let text = "a".repeat(800);
        let chunks = service.chunk_text(&text, None, None);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_text_splits_long_text_with_overlap() {
        let service = service_with(Settings::default());
        let text = "a".repeat(50);
        let chunks = service.chunk_text(&text, Some(20), Some(5));
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.len() <= 20);
        }
    }

    #[test]
    fn test_chunk_text_no_overlap_exact_division() {
        let service = service_with(Settings::default());
        let text = "0123456789".repeat(5);
        let chunks = service.chunk_text(&text, Some(10), Some(0));
        assert_eq!(chunks.len(), 5);
    }

    #[test]
    fn test_chunk_text_trims_whitespace() {
        let service = service_with(Settings::default());
        let chunks = service.chunk_text("   padded text   ", None, None);
        assert_eq!(chunks, vec!["padded text".to_string()]);
    }

    // ---- add_document ----

    #[tokio::test]
    async fn test_add_document_empty_chunks_returns_empty_without_any_request() {
        // No mock server registered at all - if this made an HTTP call it
        // would error, proving the empty-chunks short-circuit works.
        let service = service_with(settings_pointing_at("http://127.0.0.1:1"));
        let result = service.add_document(&[], "source", vec![], None).await;
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_add_document_creates_collection_when_missing_then_upserts() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(404)
            .create_async()
            .await;
        let _create = server
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .with_body(r#"{"result": true, "status": "ok"}"#)
            .create_async()
            .await;
        let _upsert = server
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+/points".to_string()))
            .with_status(200)
            .with_body(r#"{"result": {"status": "completed"}, "status": "ok"}"#)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let chunks = vec!["chunk1".to_string(), "chunk2".to_string()];
        let embeddings = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

        let result = service
            .add_document(&chunks, "source.txt", embeddings, None)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_add_document_skips_creation_when_collection_exists() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _upsert = server
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+/points".to_string()))
            .with_status(200)
            .with_body(r#"{"result": {"status": "completed"}, "status": "ok"}"#)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let chunks = vec!["chunk1".to_string()];
        let embeddings = vec![vec![0.1]];
        let mut metadata = HashMap::new();
        metadata.insert("k".to_string(), "v".to_string());

        let result = service
            .add_document(&chunks, "source.txt", embeddings, Some(metadata))
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_add_document_without_embeddings_is_validation_error() {
        let service = service_with(settings_pointing_at("http://127.0.0.1:1"));
        let chunks = vec!["chunk1".to_string()];
        let result = service.add_document(&chunks, "source.txt", vec![], None).await;
        assert!(matches!(result, Err(AppError::ValidationError(_))));
    }

    #[tokio::test]
    async fn test_add_document_upsert_failure_propagates() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _upsert = server
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+/points".to_string()))
            .with_status(500)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let result = service
            .add_document(&["c".to_string()], "s.txt", vec![vec![0.1]], None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_add_document_collection_creation_failure_propagates() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(404)
            .create_async()
            .await;
        let _create = server
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(500)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let result = service
            .add_document(&["c".to_string()], "s.txt", vec![vec![0.1]], None)
            .await;
        assert!(result.is_err());
    }

    // ---- search ----

    #[tokio::test]
    async fn test_search_returns_empty_when_collection_missing() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(404)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let result = service.search(vec![0.1, 0.2], 5).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_search_maps_results_with_score_and_metadata() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _search = server
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/search".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"result": [{"id": "abc-123", "score": 0.87, "payload": {"source": "doc.txt", "content": "hello world", "metadata": {"author": "me"}}}]}"#,
            )
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let results = service.search(vec![1.0, 0.0], 5).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "abc-123");
        assert_eq!(results[0].source, "doc.txt");
        assert_eq!(results[0].content, "hello world");
        assert_eq!(results[0].distance, Some(0.87));
        assert_eq!(results[0].metadata.get("author").unwrap(), "me");
    }

    #[tokio::test]
    async fn test_search_upstream_failure_propagates() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _search = server
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/search".to_string()))
            .with_status(500)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let result = service.search(vec![1.0], 5).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_malformed_response_propagates() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _search = server
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/search".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("not json")
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let result = service.search(vec![1.0], 5).await;
        assert!(result.is_err());
    }

    // ---- list_documents ----

    #[tokio::test]
    async fn test_list_documents_empty_when_collection_missing() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(404)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let docs = service.list_documents().await.unwrap();
        assert!(docs.is_empty());
    }

    #[tokio::test]
    async fn test_list_documents_maps_scroll_results() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _scroll = server
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/scroll".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"result": {"points": [{"id": "p1", "payload": {"source": "doc.txt", "content": "some content"}}]}}"#,
            )
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let docs = service.list_documents().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].0, "p1");
        assert_eq!(docs[0].1, "doc.txt");
        assert_eq!(docs[0].2, "some content".len());
        assert_eq!(docs[0].3, "some content");
    }

    #[tokio::test]
    async fn test_list_documents_upstream_failure_propagates() {
        let mut server = mockito::Server::new_async().await;
        let _check = server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _scroll = server
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/scroll".to_string()))
            .with_status(500)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        let result = service.list_documents().await;
        assert!(result.is_err());
    }

    // ---- clear ----

    #[tokio::test]
    async fn test_clear_success() {
        let mut server = mockito::Server::new_async().await;
        let _delete = server
            .mock("DELETE", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        assert!(service.clear().await.is_ok());
    }

    #[tokio::test]
    async fn test_clear_missing_collection_is_still_ok() {
        let mut server = mockito::Server::new_async().await;
        let _delete = server
            .mock("DELETE", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(404)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        assert!(service.clear().await.is_ok());
    }

    #[tokio::test]
    async fn test_clear_real_failure_propagates() {
        let mut server = mockito::Server::new_async().await;
        let _delete = server
            .mock("DELETE", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(500)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        assert!(service.clear().await.is_err());
    }

    // ---- api-key header ----

    #[tokio::test]
    async fn test_api_key_header_sent_when_configured() {
        let mut server = mockito::Server::new_async().await;
        let _delete = server
            .mock("DELETE", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .match_header("api-key", "secret-key")
            .with_status(200)
            .create_async()
            .await;

        let mut settings = settings_pointing_at(&server.url());
        settings.qdrant_api_key = Some("secret-key".to_string());
        let service = service_with(settings);

        assert!(service.clear().await.is_ok());
    }

    #[tokio::test]
    async fn test_no_api_key_header_when_not_configured() {
        let mut server = mockito::Server::new_async().await;
        let _delete = server
            .mock("DELETE", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .match_header("api-key", mockito::Matcher::Missing)
            .with_status(200)
            .create_async()
            .await;

        let service = service_with(settings_pointing_at(&server.url()));
        assert!(service.clear().await.is_ok());
    }

    // ---- point_id ----

    #[test]
    fn test_point_id_string_variant() {
        let point = json!({"id": "abc-123"});
        assert_eq!(VectorStoreService::point_id(&point), "abc-123");
    }

    #[test]
    fn test_point_id_numeric_fallback_variant() {
        // Qdrant also allows unsigned-integer point ids; our own ingestion
        // path always uses UUID strings, but a point stored some other way
        // (or a future numeric-id scheme) should still stringify sanely.
        let point = json!({"id": 42});
        assert_eq!(VectorStoreService::point_id(&point), "42");
    }

    // ---- new() / new_with_timeout() health-check branches ----

    #[tokio::test]
    async fn test_new_health_check_success_branch() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/healthz")
            .with_status(200)
            .create_async()
            .await;

        let result = VectorStoreService::new(settings_pointing_at(&server.url())).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_new_connection_refused_branch() {
        let settings = settings_pointing_at("http://127.0.0.1:1");
        let result =
            VectorStoreService::new_with_timeout(settings, Duration::from_millis(300)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_new_health_check_timeout_branch() {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                std::mem::forget(socket);
            }
        });

        let settings = settings_pointing_at(&format!("http://{}", addr));
        let result =
            VectorStoreService::new_with_timeout(settings, Duration::from_millis(100)).await;
        assert!(result.is_ok());
    }
}
