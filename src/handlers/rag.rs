use crate::config::Settings;
use crate::error::AppError;
use crate::models::{ClearResponse, IngestTextRequest, RAGQueryRequest};
use crate::services::RAGService;
use actix_web::{delete, get, post, web, HttpResponse};

#[post("/rag/ingest")]
pub async fn ingest_document(
    request: web::Json<IngestTextRequest>,
    settings: web::Data<Settings>,
) -> Result<HttpResponse, AppError> {
    tracing::info!(
        content_len = request.content.len(),
        source = request.source,
        "RAG ingest request"
    );

    let service = RAGService::new(settings.get_ref().clone()).await?;

    let source = if request.source.is_empty() {
        "direct_input"
    } else {
        &request.source
    };

    let response = service
        .ingest_text(&request.content, source, Some(request.metadata.clone()))
        .await?;

    tracing::debug!(chunks_count = response.chunks_count, "RAG ingest completed");
    Ok(HttpResponse::Created().json(response))
}

#[post("/rag/query")]
pub async fn query_knowledge_base(
    request: web::Json<RAGQueryRequest>,
    settings: web::Data<Settings>,
) -> Result<HttpResponse, AppError> {
    tracing::info!(
        question_len = request.question.len(),
        top_k = request.top_k,
        "RAG query request"
    );

    let service = RAGService::new(settings.get_ref().clone()).await?;

    let response = service
        .query(
            &request.question,
            request.top_k as usize,
            request.system_instruction.as_deref(),
        )
        .await?;

    tracing::debug!(result_len = response.answer.len(), "RAG query completed");
    Ok(HttpResponse::Ok().json(response))
}

#[get("/rag/documents")]
pub async fn list_indexed_documents(
    settings: web::Data<Settings>,
) -> Result<HttpResponse, AppError> {
    tracing::debug!("Listing indexed documents");

    let service = RAGService::new(settings.get_ref().clone()).await?;
    let response = service.list_documents().await?;

    tracing::debug!(count = response.documents.len(), "Documents listed");
    Ok(HttpResponse::Ok().json(response))
}

#[delete("/rag/documents")]
pub async fn clear_documents(settings: web::Data<Settings>) -> Result<HttpResponse, AppError> {
    tracing::warn!("Clearing all indexed documents");

    let service = RAGService::new(settings.get_ref().clone()).await?;
    service.clear().await?;

    tracing::info!("Knowledge base cleared");
    Ok(HttpResponse::Ok().json(ClearResponse {
        status: "success".to_string(),
        message: "Knowledge base cleared.".to_string(),
    }))
}
