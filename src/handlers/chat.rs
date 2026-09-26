use crate::error::AppError;
use crate::models::{ChatCompletionRequest, ChatCompletionResponse};
use crate::services::GeminiService;
use actix_web::{post, web, HttpResponse};
use serde_json::json;

#[post("/chat/generate")]
pub async fn generate_chat(
    request: web::Json<ChatCompletionRequest>,
    service: web::Data<GeminiService>,
) -> Result<HttpResponse, AppError> {
    tracing::info!(prompt_len = request.prompt.len(), "Chat generate request");

    let text = service
        .generate_text(
            &request.prompt,
            request.system_instruction.as_deref(),
            request.model.as_deref(),
            request.temperature,
        )
        .await?;

    tracing::debug!(response_len = text.len(), "Chat generate completed");

    Ok(HttpResponse::Ok().json(ChatCompletionResponse {
        response: text,
        model: request
            .model
            .clone()
            .unwrap_or_else(|| "gemini-2.0-flash".to_string()),
    }))
}

#[post("/chat/stream")]
pub async fn stream_chat(
    request: web::Json<ChatCompletionRequest>,
    service: web::Data<GeminiService>,
) -> Result<HttpResponse, AppError> {
    tracing::info!(prompt_len = request.prompt.len(), "Chat stream request");

    let text = service
        .generate_text(
            &request.prompt,
            request.system_instruction.as_deref(),
            request.model.as_deref(),
            request.temperature,
        )
        .await?;

    let event_stream = text
        .chars()
        .collect::<Vec<_>>()
        .chunks(20)
        .map(|window| {
            let chunk: String = window.iter().collect();
            format!(
                "data: {}\n\n",
                serde_json::to_string(&json!({
                    "text": chunk,
                    "done": false
                }))
                .unwrap_or_default()
            )
        })
        .collect::<Vec<_>>();

    let stream_data = event_stream.join("");
    let body = format!(
        "{}data: {}\n\n",
        stream_data,
        serde_json::to_string(&json!({
            "text": "",
            "done": true
        }))
        .unwrap_or_default()
    );

    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .insert_header(("X-Accel-Buffering", "no"))
        .body(body))
}
