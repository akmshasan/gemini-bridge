use crate::config::Settings;
use crate::models::{HealthResponse, ReadyResponse, RootResponse};
use actix_web::{get, web, HttpResponse};

#[get("/")]
pub async fn root(settings: web::Data<Settings>) -> HttpResponse {
    HttpResponse::Ok().json(RootResponse {
        message: format!("Welcome to {}!", settings.app_name),
        version: settings.version.clone(),
        docs: "/docs".to_string(),
        health: "/health".to_string(),
        ready: "/ready".to_string(),
    })
}

#[get("/health")]
pub async fn health_check() -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
    })
}

#[get("/ready")]
pub async fn ready_check(settings: web::Data<Settings>) -> HttpResponse {
    let has_key = !settings.gemini_api_key.is_empty();
    HttpResponse::Ok().json(ReadyResponse {
        status: if has_key { "ready" } else { "not_configured" }.to_string(),
        gemini_api_key_configured: has_key.to_string(),
    })
}
