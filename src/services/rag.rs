use crate::config::Settings;
use crate::error::AppResult;
use crate::models::{DocumentListItem, DocumentListResponse, IngestResponse, RAGQueryResponse};
use crate::services::gemini::GeminiService;
use crate::services::vector_store::VectorStoreService;
use std::collections::HashMap;

pub struct RAGService {
    gemini: GeminiService,
    vector_store: VectorStoreService,
    settings: Settings,
}

impl RAGService {
    pub async fn new(settings: Settings) -> AppResult<Self> {
        let gemini = GeminiService::new(&settings)?;
        let vector_store = VectorStoreService::new(settings.clone()).await?;

        Ok(Self {
            gemini,
            vector_store,
            settings,
        })
    }

    pub async fn ingest_text(
        &self,
        content: &str,
        source: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> AppResult<IngestResponse> {
        let chunks = self.vector_store.chunk_text(content, None, None);

        if chunks.is_empty() {
            return Ok(IngestResponse {
                source: source.to_string(),
                chunks_count: 0,
                chunk_ids: Vec::new(),
                message: "No content to ingest.".to_string(),
            });
        }

        let chunk_strings: Vec<String> = chunks.iter().cloned().collect();
        let embeddings = self.gemini.get_embeddings(&chunk_strings, None).await?;

        let chunk_ids = self
            .vector_store
            .add_document(&chunks, source, embeddings, metadata)
            .await?;

        Ok(IngestResponse {
            source: source.to_string(),
            chunks_count: chunks.len(),
            chunk_ids,
            message: format!(
                "Successfully ingested {} chunks from source '{}'.",
                chunks.len(),
                source
            ),
        })
    }

    pub async fn query(
        &self,
        question: &str,
        top_k: usize,
        system_instruction: Option<&str>,
    ) -> AppResult<RAGQueryResponse> {
        let query_embedding = self.gemini.get_embedding(question, None).await?;
        let chunks = self.vector_store.search(query_embedding, top_k).await?;

        if chunks.is_empty() {
            let fallback_prompt = format!(
                "Question: {}\n\nNote: No indexed documents were found in the knowledge base.",
                question
            );
            let answer = self
                .gemini
                .generate_text(&fallback_prompt, system_instruction, None, None)
                .await?;

            return Ok(RAGQueryResponse {
                question: question.to_string(),
                answer,
                context_chunks: Vec::new(),
                model: self.settings.gemini_model.clone(),
            });
        }

        let mut context_blocks = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            context_blocks.push(format!(
                "[{}] Source: {}\n{}",
                idx + 1,
                chunk.source,
                chunk.content
            ));
        }
        let joined_context = context_blocks.join("\n\n");

        let augmented_prompt = format!(
            "You are a helpful assistant answering questions using the retrieved reference context below.\n\n\
             === RETRIEVED CONTEXT ===\n\
             {}\n\
             =========================\n\n\
             Question: {}\n\n\
             Instructions: Provide a clear, accurate answer based on the context above. Cite sources [1], [2], etc.",
            joined_context, question
        );

        let answer = self
            .gemini
            .generate_text(&augmented_prompt, system_instruction, None, None)
            .await?;

        Ok(RAGQueryResponse {
            question: question.to_string(),
            answer,
            context_chunks: chunks,
            model: self.settings.gemini_model.clone(),
        })
    }

    pub async fn list_documents(&self) -> AppResult<DocumentListResponse> {
        let docs = self.vector_store.list_documents().await?;

        let documents: Vec<DocumentListItem> = docs
            .into_iter()
            .map(|(chunk_id, source, char_length, preview)| DocumentListItem {
                chunk_id,
                source,
                char_length,
                preview,
            })
            .collect();

        Ok(DocumentListResponse {
            total_chunks: documents.len(),
            documents,
        })
    }

    pub async fn clear(&self) -> AppResult<()> {
        self.vector_store.clear().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_settings(gemini_base_url: &str, qdrant_url: &str) -> Settings {
        let mut s = Settings::default();
        s.gemini_api_key = "test-key".to_string();
        s.gemini_api_base_url = gemini_base_url.to_string();
        s.qdrant_url = qdrant_url.to_string();
        s
    }

    /// A Qdrant mock where the collection always reports as already
    /// existing (so creation is skipped) and every read comes back empty -
    /// enough for a full ingest-then-query flow to run against a mock
    /// server rather than a real Qdrant instance.
    async fn mock_qdrant_server() -> mockito::ServerGuard {
        let mut server = mockito::Server::new_async().await;

        server
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        server
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .with_body(r#"{"result": true, "status": "ok"}"#)
            .create_async()
            .await;
        server
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+/points".to_string()))
            .with_status(200)
            .with_body(r#"{"result": {"status": "completed"}, "status": "ok"}"#)
            .create_async()
            .await;
        server
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/search".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"result": []}"#)
            .create_async()
            .await;
        server
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/scroll".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"result": {"points": []}}"#)
            .create_async()
            .await;
        server
            .mock("DELETE", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;

        server
    }

    #[tokio::test]
    async fn test_new_fails_without_gemini_key() {
        let qdrant = mock_qdrant_server().await;
        let mut settings = base_settings("http://127.0.0.1:1", &qdrant.url());
        settings.gemini_api_key = String::new();

        let result = RAGService::new(settings).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_new_succeeds_with_key() {
        let qdrant = mock_qdrant_server().await;
        let settings = base_settings("http://127.0.0.1:1", &qdrant.url());
        let result = RAGService::new(settings).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ingest_text_empty_content_short_circuits() {
        let qdrant = mock_qdrant_server().await;
        let settings = base_settings("http://127.0.0.1:1", &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        let result = service.ingest_text("   ", "empty.txt", None).await.unwrap();
        assert_eq!(result.chunks_count, 0);
        assert!(result.chunk_ids.is_empty());
        assert_eq!(result.message, "No content to ingest.");
    }

    #[tokio::test]
    async fn test_ingest_text_full_flow_with_mocked_embeddings() {
        let mut gemini = mockito::Server::new_async().await;
        let _mock = gemini
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding":{"values":[0.1,0.2]}}"#)
            .create_async()
            .await;
        let qdrant = mock_qdrant_server().await;

        let settings = base_settings(&gemini.url(), &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        let result = service
            .ingest_text("Some content to ingest about Rust.", "doc.txt", None)
            .await
            .unwrap();

        assert_eq!(result.source, "doc.txt");
        assert!(result.chunks_count > 0);
        assert_eq!(result.chunk_ids.len(), result.chunks_count);
    }

    #[tokio::test]
    async fn test_ingest_text_propagates_embedding_error() {
        // Nothing listening -> get_embeddings() fails, error should propagate.
        let qdrant = mock_qdrant_server().await;
        let settings = base_settings("http://127.0.0.1:1", &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        let result = service
            .ingest_text("Some content", "doc.txt", None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_query_with_no_indexed_documents_uses_fallback() {
        let mut gemini = mockito::Server::new_async().await;
        let _embed = gemini
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding":{"values":[0.1,0.2]}}"#)
            .create_async()
            .await;
        let _generate = gemini
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"No docs found answer"}]}}]}"#,
            )
            .create_async()
            .await;
        let qdrant = mock_qdrant_server().await;

        let settings = base_settings(&gemini.url(), &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        let result = service.query("What is Rust?", 5, None).await.unwrap();
        assert_eq!(result.answer, "No docs found answer");
        assert!(result.context_chunks.is_empty());
        assert_eq!(result.question, "What is Rust?");
    }

    #[tokio::test]
    async fn test_query_with_system_instruction() {
        let mut gemini = mockito::Server::new_async().await;
        let _embed = gemini
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding":{"values":[0.1]}}"#)
            .create_async()
            .await;
        let _generate = gemini
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"answer"}]}}]}"#)
            .create_async()
            .await;
        let qdrant = mock_qdrant_server().await;

        let settings = base_settings(&gemini.url(), &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        let result = service
            .query("Question?", 3, Some("Answer briefly"))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_query_with_indexed_documents_uses_retrieved_context() {
        let mut gemini = mockito::Server::new_async().await;
        let _embed = gemini
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding":{"values":[1.0,0.0]}}"#)
            .create_async()
            .await;
        let _generate = gemini
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Cited answer [1]"}]}}]}"#,
            )
            .create_async()
            .await;

        // Unlike mock_qdrant_server(), this test needs search to actually
        // return the ingested chunk, so it wires the collection/search mocks
        // by hand instead of using the always-empty default.
        let mut qdrant = mockito::Server::new_async().await;
        let _check = qdrant
            .mock("GET", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _create = qdrant
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()))
            .with_status(200)
            .create_async()
            .await;
        let _upsert = qdrant
            .mock("PUT", mockito::Matcher::Regex(r"^/collections/[^/]+/points".to_string()))
            .with_status(200)
            .with_body(r#"{"result": {"status": "completed"}, "status": "ok"}"#)
            .create_async()
            .await;
        let _search = qdrant
            .mock("POST", mockito::Matcher::Regex(r"^/collections/[^/]+/points/search".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"result": [{"id": "rust-1", "score": 0.98, "payload": {"source": "rust.txt", "content": "Rust is a memory-safe systems language."}}]}"#,
            )
            .create_async()
            .await;

        let settings = base_settings(&gemini.url(), &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        // Ingest first so the store has something to retrieve on query.
        service
            .ingest_text("Rust is a memory-safe systems language.", "rust.txt", None)
            .await
            .unwrap();

        let result = service.query("What is Rust?", 5, None).await.unwrap();

        assert_eq!(result.answer, "Cited answer [1]");
        assert!(!result.context_chunks.is_empty());
        assert_eq!(result.context_chunks[0].source, "rust.txt");
    }

    #[tokio::test]
    async fn test_query_propagates_embedding_error() {
        let qdrant = mock_qdrant_server().await;
        let settings = base_settings("http://127.0.0.1:1", &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        let result = service.query("Question?", 5, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_documents_empty() {
        let qdrant = mock_qdrant_server().await;
        let settings = base_settings("http://127.0.0.1:1", &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        let result = service.list_documents().await.unwrap();
        assert_eq!(result.total_chunks, 0);
        assert!(result.documents.is_empty());
    }

    #[tokio::test]
    async fn test_clear_succeeds() {
        let qdrant = mock_qdrant_server().await;
        let settings = base_settings("http://127.0.0.1:1", &qdrant.url());
        let service = RAGService::new(settings).await.unwrap();

        assert!(service.clear().await.is_ok());
    }
}
