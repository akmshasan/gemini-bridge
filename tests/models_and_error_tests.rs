use gemini_bridge::models::*;
use std::collections::HashMap;

// ============================================================================
// MODELS TESTS - construction + serde round-trips for every struct
//
// AppError coverage lives in src/error/mod.rs's own #[cfg(test)] module,
// alongside the status-code mapping it verifies.
// ============================================================================

#[test]
fn test_chat_completion_request_roundtrip() {
    let req = ChatCompletionRequest {
        prompt: "What is Rust?".to_string(),
        system_instruction: Some("Be helpful".to_string()),
        model: Some("gemini-3.8-flash".to_string()),
        temperature: Some(0.7),
    };
    let json = serde_json::to_string(&req).unwrap();
    let decoded: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.prompt, "What is Rust?");
    assert_eq!(decoded.temperature, Some(0.7));
}

#[test]
fn test_chat_completion_request_defaults_missing_optionals() {
    let json = r#"{"prompt": "Hi"}"#;
    let decoded: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(decoded.prompt, "Hi");
    assert!(decoded.system_instruction.is_none());
    assert!(decoded.model.is_none());
    assert!(decoded.temperature.is_none());
}

#[test]
fn test_chat_completion_response() {
    let resp = ChatCompletionResponse {
        response: "Rust is great".to_string(),
        model: "gemini-3.8-flash".to_string(),
    };
    assert_eq!(resp.response, "Rust is great");
}

#[test]
fn test_ingest_text_request_defaults() {
    let json = r#"{"content": "Document text"}"#;
    let decoded: IngestTextRequest = serde_json::from_str(json).unwrap();
    assert_eq!(decoded.content, "Document text");
    assert_eq!(decoded.source, "");
    assert!(decoded.metadata.is_empty());
}

#[test]
fn test_ingest_text_request_full() {
    let mut metadata = HashMap::new();
    metadata.insert("author".to_string(), "user".to_string());
    let req = IngestTextRequest {
        content: "Document text".to_string(),
        source: "doc.txt".to_string(),
        metadata,
    };
    let json = serde_json::to_string(&req).unwrap();
    let decoded: IngestTextRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.source, "doc.txt");
    assert_eq!(decoded.metadata.get("author").unwrap(), "user");
}

#[test]
fn test_ingest_response() {
    let resp = IngestResponse {
        source: "doc".to_string(),
        chunks_count: 5,
        chunk_ids: vec!["c1".to_string(), "c2".to_string()],
        message: "ok".to_string(),
    };
    assert_eq!(resp.chunks_count, 5);
    assert_eq!(resp.chunk_ids.len(), 2);
}

#[test]
fn test_rag_query_request_default_top_k() {
    let json = r#"{"question": "What is Rust?"}"#;
    let decoded: RAGQueryRequest = serde_json::from_str(json).unwrap();
    assert_eq!(decoded.question, "What is Rust?");
    assert_eq!(decoded.top_k, 4);
    assert!(decoded.system_instruction.is_none());
}

#[test]
fn test_rag_query_request_explicit_top_k() {
    let req = RAGQueryRequest {
        question: "What is Rust?".to_string(),
        top_k: 5,
        system_instruction: Some("Expert mode".to_string()),
    };
    let json = serde_json::to_string(&req).unwrap();
    let decoded: RAGQueryRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.top_k, 5);
}

#[test]
fn test_rag_query_response_with_chunks() {
    let chunk = DocumentChunkResult {
        chunk_id: "c1".to_string(),
        content: "Rust content".to_string(),
        source: "doc".to_string(),
        distance: Some(0.95),
        metadata: HashMap::new(),
    };
    let resp = RAGQueryResponse {
        question: "What is Rust?".to_string(),
        answer: "Rust is a language".to_string(),
        context_chunks: vec![chunk],
        model: "gemini-3.8-flash".to_string(),
    };
    assert_eq!(resp.context_chunks.len(), 1);
    assert_eq!(resp.context_chunks[0].distance, Some(0.95));
}

#[test]
fn test_document_chunk_result_without_distance() {
    let json = r#"{"chunk_id":"c1","content":"text","source":"doc"}"#;
    let decoded: DocumentChunkResult = serde_json::from_str(json).unwrap();
    assert!(decoded.distance.is_none());
    assert!(decoded.metadata.is_empty());
}

#[test]
fn test_document_list_response() {
    let item = DocumentListItem {
        chunk_id: "c1".to_string(),
        source: "doc".to_string(),
        char_length: 100,
        preview: "Preview".to_string(),
    };
    let resp = DocumentListResponse {
        total_chunks: 1,
        documents: vec![item],
    };
    assert_eq!(resp.total_chunks, 1);
}

#[test]
fn test_health_response() {
    let resp = HealthResponse {
        status: "ok".to_string(),
    };
    assert_eq!(resp.status, "ok");
}

#[test]
fn test_ready_response() {
    let resp = ReadyResponse {
        status: "ready".to_string(),
        gemini_api_key_configured: "true".to_string(),
    };
    assert_eq!(resp.status, "ready");
}

#[test]
fn test_root_response() {
    let resp = RootResponse {
        message: "Welcome".to_string(),
        version: "0.1.0".to_string(),
        docs: "/docs".to_string(),
        health: "/health".to_string(),
        ready: "/ready".to_string(),
    };
    assert_eq!(resp.version, "0.1.0");
}

#[test]
fn test_clear_response() {
    let resp = ClearResponse {
        status: "success".to_string(),
        message: "Cleared".to_string(),
    };
    assert_eq!(resp.status, "success");
}

#[test]
fn test_gemini_content_and_part() {
    let content = GeminiContent {
        role: "user".to_string(),
        parts: vec![GeminiPart {
            text: "Hello".to_string(),
        }],
    };
    assert_eq!(content.parts.len(), 1);
    assert_eq!(content.parts[0].text, "Hello");
}

#[test]
fn test_gemini_request_with_and_without_optionals() {
    let request = GeminiRequest {
        contents: vec![GeminiContent {
            role: "user".to_string(),
            parts: vec![GeminiPart {
                text: "Hi".to_string(),
            }],
        }],
        system_instruction: None,
        generation_config: None,
    };
    let json = serde_json::to_string(&request).unwrap();
    assert!(!json.contains("generation_config"));

    let request_with_config = GeminiRequest {
        contents: vec![],
        system_instruction: Some(GeminiContent {
            role: "user".to_string(),
            parts: vec![],
        }),
        generation_config: Some(GenerationConfig {
            temperature: Some(0.5),
        }),
    };
    let json2 = serde_json::to_string(&request_with_config).unwrap();
    assert!(json2.contains("generation_config"));
}

#[test]
fn test_generation_config() {
    let config = GenerationConfig {
        temperature: Some(0.7),
    };
    assert_eq!(config.temperature, Some(0.7));
}

#[test]
fn test_candidate_and_gemini_response() {
    let candidate = Candidate {
        content: GeminiContent {
            role: "assistant".to_string(),
            parts: vec![GeminiPart {
                text: "Response".to_string(),
            }],
        },
    };
    let response = GeminiResponse {
        candidates: vec![candidate],
    };
    assert_eq!(response.candidates.len(), 1);
}

#[test]
fn test_embedding_request_and_response() {
    let req = EmbeddingRequest {
        model: "embedding-model".to_string(),
        content: GeminiContent {
            role: "user".to_string(),
            parts: vec![],
        },
    };
    assert_eq!(req.model, "embedding-model");

    let resp = EmbeddingResponse {
        embedding: Embedding {
            values: vec![0.1, 0.2],
        },
    };
    assert_eq!(resp.embedding.values.len(), 2);
}

#[test]
fn test_embedding_struct() {
    let embedding = Embedding {
        values: vec![0.1, 0.2, 0.3, 0.4, 0.5],
    };
    assert_eq!(embedding.values.len(), 5);
}
