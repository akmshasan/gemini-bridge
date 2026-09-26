//! Gemini Bridge - Application Entry Point

use actix_web::{middleware, web, App, HttpServer};
use gemini_bridge::{config::get_settings, handlers, services::GeminiService};
use tracing_subscriber;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize tracing with JSON output
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .json()
        .init();

    let settings = get_settings();

    tracing::info!(
        name = settings.app_name,
        version = settings.version,
        server = format!("{}:{}", settings.server_host, settings.server_port),
        qdrant = settings.qdrant_url,
        "Starting Gemini Bridge"
    );

    let host = settings.server_host.clone();
    let port = settings.server_port;
    let settings_data = web::Data::new(settings.clone());

    // Initialize Gemini service once (Client is reused)
    let gemini_service = GeminiService::new(&settings).map_err(|e| {
        tracing::error!(error = %e, "Failed to initialize Gemini service");
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })?;
    let gemini_service_data = web::Data::new(gemini_service);

    // Start HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(settings_data.clone())
            .app_data(gemini_service_data.clone())
            .wrap(middleware::Logger::default())
            .wrap(tracing_actix_web::TracingLogger::default())
            .wrap(
                actix_web::middleware::DefaultHeaders::new()
                    .add(("X-App-Name", "Gemini Bridge"))
                    .add(("X-Version", "0.1.0")),
            )
            // Health check routes
            .service(handlers::health::root)
            .service(handlers::health::health_check)
            .service(handlers::health::ready_check)
            // Chat routes
            .service(handlers::chat::generate_chat)
            .service(handlers::chat::stream_chat)
            // RAG routes
            .service(handlers::rag::ingest_document)
            .service(handlers::rag::query_knowledge_base)
            .service(handlers::rag::list_indexed_documents)
            .service(handlers::rag::clear_documents)
    })
    .bind(format!("{}:{}", host, port))?
    .run()
    .await
    .map_err(|e| {
        tracing::error!(error = %e, "Server error");
        e
    })?;

    tracing::info!("Gemini Bridge stopped");
    Ok(())
}
