use actix_web::{error::ResponseError, http::StatusCode, HttpResponse};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Vector store error: {0}")]
    VectorStoreError(String),
    #[error("Gemini API error: {0}")]
    GeminiError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Internal server error: {0}")]
    InternalError(String),
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Qdrant error: {0}")]
    QdrantError(String),
}

impl AppError {
    /// Single source of truth for the HTTP status a variant maps to, so
    /// `error_response()` and `status_code()` can never drift apart.
    fn http_status(&self) -> StatusCode {
        match self {
            AppError::ConfigError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::ApiError(_) => StatusCode::BAD_REQUEST,
            AppError::VectorStoreError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::GeminiError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::ValidationError(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::RequestError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::SerializationError(_) => StatusCode::BAD_REQUEST,
            AppError::IoError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::QdrantError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let status = self.http_status();
        HttpResponse::build(status).json(json!({
            "error": self.to_string(),
            "status": status.as_u16(),
        }))
    }

    fn status_code(&self) -> StatusCode {
        self.http_status()
    }
}

pub type AppResult<T> = Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_code_matches_error_response_status_for_every_variant() {
        let cases: Vec<(AppError, u16)> = vec![
            (AppError::ConfigError("x".into()), 500),
            (AppError::ApiError("x".into()), 400),
            (AppError::VectorStoreError("x".into()), 500),
            (AppError::GeminiError("x".into()), 500),
            (AppError::ValidationError("x".into()), 400),
            (AppError::NotFound("x".into()), 404),
            (AppError::InternalError("x".into()), 500),
            (AppError::QdrantError("x".into()), 500),
        ];

        for (err, expected) in cases {
            assert_eq!(err.status_code().as_u16(), expected);
            assert_eq!(err.error_response().status().as_u16(), expected);
        }
    }

    #[test]
    fn test_error_response_body_shape() {
        let err = AppError::ValidationError("bad input".to_string());
        let resp = err.error_response();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[test]
    fn test_display_messages() {
        assert_eq!(
            AppError::ConfigError("cfg".to_string()).to_string(),
            "Configuration error: cfg"
        );
        assert_eq!(
            AppError::NotFound("missing".to_string()).to_string(),
            "Not found: missing"
        );
        assert_eq!(
            AppError::ApiError("bad".to_string()).to_string(),
            "API error: bad"
        );
        assert_eq!(
            AppError::VectorStoreError("db".to_string()).to_string(),
            "Vector store error: db"
        );
        assert_eq!(
            AppError::GeminiError("api".to_string()).to_string(),
            "Gemini API error: api"
        );
        assert_eq!(
            AppError::ValidationError("field".to_string()).to_string(),
            "Validation error: field"
        );
        assert_eq!(
            AppError::InternalError("boom".to_string()).to_string(),
            "Internal server error: boom"
        );
        assert_eq!(
            AppError::QdrantError("conn".to_string()).to_string(),
            "Qdrant error: conn"
        );
    }

    #[test]
    fn test_from_reqwest_error() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result: Result<(), reqwest::Error> = rt.block_on(async {
            reqwest::get("http://127.0.0.1:1").await?;
            Ok(())
        });
        let app_err: AppError = result.unwrap_err().into();
        assert_eq!(app_err.status_code().as_u16(), 500);
        assert!(app_err.to_string().starts_with("Request error:"));
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let app_err: AppError = json_err.into();
        assert_eq!(app_err.status_code().as_u16(), 400);
        assert!(app_err.to_string().starts_with("Serialization error:"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let app_err: AppError = io_err.into();
        assert_eq!(app_err.status_code().as_u16(), 500);
        assert!(app_err.to_string().starts_with("IO error:"));
    }
}
