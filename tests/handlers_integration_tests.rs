use actix_web::{test, web, App};
use gemini_bridge::config::Settings;
use gemini_bridge::handlers;
use gemini_bridge::models::*;
use gemini_bridge::services::GeminiService;
use std::collections::HashMap;

fn test_settings(gemini_base_url: &str, api_key: &str, qdrant_url: &str) -> Settings {
    let mut settings = Settings::default();
    settings.gemini_api_key = api_key.to_string();
    settings.gemini_api_base_url = gemini_base_url.to_string();
    settings.qdrant_url = qdrant_url.to_string();
    settings
}

/// A Qdrant mock where the collection always reports as already existing
/// (so creation is skipped) and every read comes back empty - enough for
/// handler tests that exercise the RAG endpoints end-to-end without a real
/// Qdrant instance.
async fn mock_qdrant_server() -> mockito::ServerGuard {
    let mut server = mockito::Server::new_async().await;

    server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()),
        )
        .with_status(200)
        .create_async()
        .await;
    server
        .mock(
            "PUT",
            mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()),
        )
        .with_status(200)
        .with_body(r#"{"result": true, "status": "ok"}"#)
        .create_async()
        .await;
    server
        .mock(
            "PUT",
            mockito::Matcher::Regex(r"^/collections/[^/]+/points".to_string()),
        )
        .with_status(200)
        .with_body(r#"{"result": {"status": "completed"}, "status": "ok"}"#)
        .create_async()
        .await;
    server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/collections/[^/]+/points/search".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"result": []}"#)
        .create_async()
        .await;
    server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/collections/[^/]+/points/scroll".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"result": {"points": []}}"#)
        .create_async()
        .await;
    server
        .mock(
            "DELETE",
            mockito::Matcher::Regex(r"^/collections/[^/]+$".to_string()),
        )
        .with_status(200)
        .create_async()
        .await;

    server
}

// ============================================================================
// Health handlers
// ============================================================================

#[actix_web::test]
async fn test_root_endpoint() {
    let settings = test_settings("http://127.0.0.1:1", "", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::health::root),
    )
    .await;

    let req = test::TestRequest::get().uri("/").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: RootResponse = test::read_body_json(resp).await;
    assert!(body.message.contains("Gemini Bridge"));
    assert_eq!(body.health, "/health");
    assert_eq!(body.ready, "/ready");
}

#[actix_web::test]
async fn test_health_endpoint() {
    let app = test::init_service(App::new().service(handlers::health::health_check)).await;
    let req = test::TestRequest::get().uri("/health").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    let body: HealthResponse = test::read_body_json(resp).await;
    assert_eq!(body.status, "ok");
}

#[actix_web::test]
async fn test_ready_endpoint_configured() {
    let settings = test_settings("http://127.0.0.1:1", "test-key", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::health::ready_check),
    )
    .await;
    let req = test::TestRequest::get().uri("/ready").to_request();
    let resp = test::call_service(&app, req).await;
    let body: ReadyResponse = test::read_body_json(resp).await;
    assert_eq!(body.status, "ready");
    assert_eq!(body.gemini_api_key_configured, "true");
}

#[actix_web::test]
async fn test_ready_endpoint_not_configured() {
    let settings = test_settings("http://127.0.0.1:1", "", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::health::ready_check),
    )
    .await;
    let req = test::TestRequest::get().uri("/ready").to_request();
    let resp = test::call_service(&app, req).await;
    let body: ReadyResponse = test::read_body_json(resp).await;
    assert_eq!(body.status, "not_configured");
    assert_eq!(body.gemini_api_key_configured, "false");
}

#[actix_web::test]
async fn test_404_on_nonexistent_route() {
    let settings = test_settings("http://127.0.0.1:1", "test-key", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::health::root),
    )
    .await;

    let req = test::TestRequest::get().uri("/nonexistent").to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 404);
}

// ============================================================================
// Chat handlers
// ============================================================================

#[actix_web::test]
async fn test_chat_generate_success() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi there!"}]}}]}"#)
        .create_async()
        .await;

    let settings = test_settings(&server.url(), "test-key", "http://127.0.0.1:1");
    let service = GeminiService::new(&settings).expect("Failed to create GeminiService");

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(service))
            .service(handlers::chat::generate_chat),
    )
    .await;

    let payload = ChatCompletionRequest {
        prompt: "Hello".to_string(),
        system_instruction: None,
        model: None,
        temperature: None,
    };

    let req = test::TestRequest::post()
        .uri("/chat/generate")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    let body: ChatCompletionResponse = test::read_body_json(resp).await;
    assert_eq!(body.response, "Hi there!");
}

#[actix_web::test]
async fn test_chat_generate_without_api_key_returns_500() {
    // Use a dummy key to allow service creation; the 500 error will occur on API call
    let settings = test_settings("http://127.0.0.1:1", "dummy-key", "http://127.0.0.1:1");
    let service = GeminiService::new(&settings).expect("Failed to create GeminiService");

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(service))
            .service(handlers::chat::generate_chat),
    )
    .await;

    let payload = ChatCompletionRequest {
        prompt: "Hello".to_string(),
        system_instruction: None,
        model: None,
        temperature: None,
    };

    let req = test::TestRequest::post()
        .uri("/chat/generate")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 500);
}

#[actix_web::test]
async fn test_chat_generate_upstream_error_returns_500() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
        )
        .with_status(400)
        .create_async()
        .await;

    let settings = test_settings(&server.url(), "test-key", "http://127.0.0.1:1");
    let service = GeminiService::new(&settings).expect("Failed to create GeminiService");

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(service))
            .service(handlers::chat::generate_chat),
    )
    .await;

    let payload = ChatCompletionRequest {
        prompt: "Hello".to_string(),
        system_instruction: None,
        model: None,
        temperature: None,
    };

    let req = test::TestRequest::post()
        .uri("/chat/generate")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 500);
}

#[actix_web::test]
async fn test_chat_generate_invalid_json_body_returns_400() {
    let settings = test_settings("http://127.0.0.1:1", "test-key", "http://127.0.0.1:1");
    let service = GeminiService::new(&settings).expect("Failed to create GeminiService");

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(service))
            .service(handlers::chat::generate_chat),
    )
    .await;

    let req = test::TestRequest::post()
        .uri("/chat/generate")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(r#"{"invalid": "json""#)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_client_error());
}

#[actix_web::test]
async fn test_chat_stream_success() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Streaming response text"}]}}]}"#,
        )
        .create_async()
        .await;

    let settings = test_settings(&server.url(), "test-key", "http://127.0.0.1:1");
    let service = GeminiService::new(&settings).expect("Failed to create GeminiService");

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(service))
            .service(handlers::chat::stream_chat),
    )
    .await;

    let payload = ChatCompletionRequest {
        prompt: "Hello".to_string(),
        system_instruction: None,
        model: None,
        temperature: None,
    };

    let req = test::TestRequest::post()
        .uri("/chat/stream")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
}

#[actix_web::test]
async fn test_chat_stream_without_api_key_returns_500() {
    // Use a dummy key to allow service creation; the 500 error will occur on API call
    let settings = test_settings("http://127.0.0.1:1", "dummy-key", "http://127.0.0.1:1");
    let service = GeminiService::new(&settings).expect("Failed to create GeminiService");

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(service))
            .service(handlers::chat::stream_chat),
    )
    .await;

    let payload = ChatCompletionRequest {
        prompt: "Hello".to_string(),
        system_instruction: None,
        model: None,
        temperature: None,
    };

    let req = test::TestRequest::post()
        .uri("/chat/stream")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 500);
}

// ============================================================================
// RAG handlers
// ============================================================================

#[actix_web::test]
async fn test_rag_ingest_success() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
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

    let settings = test_settings(&server.url(), "test-key", &qdrant.url());
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::ingest_document),
    )
    .await;

    let payload = IngestTextRequest {
        content: "Some document content".to_string(),
        source: "test.txt".to_string(),
        metadata: HashMap::new(),
    };

    let req = test::TestRequest::post()
        .uri("/rag/ingest")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 201);
    let body: IngestResponse = test::read_body_json(resp).await;
    assert_eq!(body.source, "test.txt");
}

#[actix_web::test]
async fn test_rag_ingest_defaults_source_when_empty() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding":{"values":[0.1]}}"#)
        .create_async()
        .await;
    let qdrant = mock_qdrant_server().await;

    let settings = test_settings(&server.url(), "test-key", &qdrant.url());
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::ingest_document),
    )
    .await;

    let payload = IngestTextRequest {
        content: "Content without explicit source".to_string(),
        source: String::new(),
        metadata: HashMap::new(),
    };

    let req = test::TestRequest::post()
        .uri("/rag/ingest")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 201);
    let body: IngestResponse = test::read_body_json(resp).await;
    assert_eq!(body.source, "direct_input");
}

#[actix_web::test]
async fn test_rag_ingest_missing_content_field_returns_400() {
    // Body deserialization fails before the handler ever reaches Qdrant.
    let settings = test_settings("http://127.0.0.1:1", "test-key", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::ingest_document),
    )
    .await;

    let req = test::TestRequest::post()
        .uri("/rag/ingest")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(r#"{"source": "test.txt"}"#)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_client_error());
}

#[actix_web::test]
async fn test_rag_ingest_without_api_key_returns_500() {
    // GeminiService::new fails before VectorStoreService is ever reached.
    let settings = test_settings("http://127.0.0.1:1", "", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::ingest_document),
    )
    .await;

    let payload = IngestTextRequest {
        content: "Content".to_string(),
        source: "test.txt".to_string(),
        metadata: HashMap::new(),
    };

    let req = test::TestRequest::post()
        .uri("/rag/ingest")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 500);
}

#[actix_web::test]
async fn test_rag_query_success() {
    let mut server = mockito::Server::new_async().await;
    let _embed = server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding":{"values":[0.1,0.2]}}"#)
        .create_async()
        .await;
    let _generate = server
        .mock(
            "POST",
            mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"The answer"}]}}]}"#)
        .create_async()
        .await;
    let qdrant = mock_qdrant_server().await;

    let settings = test_settings(&server.url(), "test-key", &qdrant.url());
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::query_knowledge_base),
    )
    .await;

    let payload = RAGQueryRequest {
        question: "What is Rust?".to_string(),
        top_k: 5,
        system_instruction: None,
    };

    let req = test::TestRequest::post()
        .uri("/rag/query")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    let body: RAGQueryResponse = test::read_body_json(resp).await;
    assert_eq!(body.answer, "The answer");
}

#[actix_web::test]
async fn test_rag_query_missing_question_returns_400() {
    // Body deserialization fails before the handler ever reaches Qdrant.
    let settings = test_settings("http://127.0.0.1:1", "test-key", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::query_knowledge_base),
    )
    .await;

    let req = test::TestRequest::post()
        .uri("/rag/query")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(r#"{"top_k": 5}"#)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_client_error());
}

#[actix_web::test]
async fn test_rag_query_without_api_key_returns_500() {
    // GeminiService::new fails before VectorStoreService is ever reached.
    let settings = test_settings("http://127.0.0.1:1", "", "http://127.0.0.1:1");
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::query_knowledge_base),
    )
    .await;

    let payload = RAGQueryRequest {
        question: "What is Rust?".to_string(),
        top_k: 5,
        system_instruction: None,
    };

    let req = test::TestRequest::post()
        .uri("/rag/query")
        .set_json(&payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 500);
}

#[actix_web::test]
async fn test_rag_documents_list() {
    let qdrant = mock_qdrant_server().await;
    let settings = test_settings("http://127.0.0.1:1", "test-key", &qdrant.url());
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::list_indexed_documents),
    )
    .await;

    let req = test::TestRequest::get().uri("/rag/documents").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    let body: DocumentListResponse = test::read_body_json(resp).await;
    assert_eq!(body.total_chunks, 0);
}

#[actix_web::test]
async fn test_rag_documents_clear() {
    let qdrant = mock_qdrant_server().await;
    let settings = test_settings("http://127.0.0.1:1", "test-key", &qdrant.url());
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::rag::clear_documents),
    )
    .await;

    let req = test::TestRequest::delete()
        .uri("/rag/documents")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    let body: ClearResponse = test::read_body_json(resp).await;
    assert_eq!(body.status, "success");
}

// ============================================================================
// Full app wiring sanity check
// ============================================================================

#[actix_web::test]
async fn test_all_routes_registered_together() {
    let qdrant = mock_qdrant_server().await;
    let settings = test_settings("http://127.0.0.1:1", "test-key", &qdrant.url());
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(settings))
            .service(handlers::health::root)
            .service(handlers::health::health_check)
            .service(handlers::health::ready_check)
            .service(handlers::chat::generate_chat)
            .service(handlers::chat::stream_chat)
            .service(handlers::rag::ingest_document)
            .service(handlers::rag::query_knowledge_base)
            .service(handlers::rag::list_indexed_documents)
            .service(handlers::rag::clear_documents),
    )
    .await;

    for (method, uri) in [
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/ready"),
        ("GET", "/rag/documents"),
    ] {
        let req = test::TestRequest::with_uri(uri)
            .method(method.parse().unwrap())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "failed for {} {}", method, uri);
    }
}
