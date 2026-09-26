use crate::config::Settings;
use crate::error::{AppError, AppResult};
use crate::models::{
    EmbeddingRequest, EmbeddingResponse, GeminiContent, GeminiPart, GeminiRequest,
    GeminiResponse, GenerationConfig,
};
use reqwest::Client;

pub struct GeminiService {
    api_key: String,
    model: String,
    embedding_model: String,
    base_url: String,
    http_client: Client,
}

impl GeminiService {
    pub fn new(settings: &Settings) -> AppResult<Self> {
        if settings.gemini_api_key.is_empty() {
            return Err(AppError::ConfigError(
                "GEMINI_API_KEY is not configured".to_string(),
            ));
        }

        Ok(Self {
            api_key: settings.gemini_api_key.clone(),
            model: settings.gemini_model.clone(),
            embedding_model: settings.gemini_embedding_model.clone(),
            base_url: settings.gemini_api_base_url.clone(),
            http_client: Client::new(),
        })
    }

    pub async fn generate_text(
        &self,
        prompt: &str,
        system_instruction: Option<&str>,
        model: Option<&str>,
        temperature: Option<f32>,
    ) -> AppResult<String> {
        let target_model = model.unwrap_or(&self.model);

        let request = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: prompt.to_string(),
                }],
            }],
            system_instruction: system_instruction.map(|s| GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: s.to_string(),
                }],
            }),
            generation_config: temperature.map(|t| GenerationConfig {
                temperature: Some(t),
            }),
        };

        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, target_model, self.api_key
        );

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AppError::GeminiError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::GeminiError(format!(
                "API returned status {}: {}",
                status, error_text
            )));
        }

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .map_err(|e| AppError::GeminiError(e.to_string()))?;

        gemini_response
            .candidates
            .first()
            .and_then(|c| c.content.parts.first().map(|p| p.text.clone()))
            .ok_or_else(|| AppError::GeminiError("No response text generated".to_string()))
    }

    pub async fn get_embedding(&self, text: &str, model: Option<&str>) -> AppResult<Vec<f32>> {
        let target_model = model.unwrap_or(&self.embedding_model);

        let request = EmbeddingRequest {
            model: format!("models/{}", target_model),
            content: GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: text.to_string(),
                }],
            },
        };

        let url = format!(
            "{}/v1beta/models/{}:embedContent?key={}",
            self.base_url, target_model, self.api_key
        );

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AppError::GeminiError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(AppError::GeminiError("Embedding API error".to_string()));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| AppError::GeminiError(e.to_string()))?;

        Ok(embedding_response.embedding.values)
    }

    pub async fn get_embeddings(
        &self,
        texts: &[String],
        model: Option<&str>,
    ) -> AppResult<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for text in texts {
            let embedding = self.get_embedding(text, model).await?;
            results.push(embedding);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings_with(api_key: &str, base_url: &str) -> Settings {
        let mut s = Settings::default();
        s.gemini_api_key = api_key.to_string();
        s.gemini_api_base_url = base_url.to_string();
        s
    }

    #[test]
    fn test_new_fails_without_api_key() {
        let settings = settings_with("", "http://localhost");
        assert!(GeminiService::new(&settings).is_err());
    }

    #[test]
    fn test_new_succeeds_with_api_key() {
        let settings = settings_with("test-key", "http://localhost");
        assert!(GeminiService::new(&settings).is_ok());
    }

    #[tokio::test]
    async fn test_generate_text_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello there"}]}}]}"#)
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service.generate_text("Hi", None, None, None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello there");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_generate_text_with_system_instruction_model_and_temperature() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Answer"}]}}]}"#)
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service
            .generate_text("Hi", Some("Be concise"), Some("custom-model"), Some(0.5))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Answer");
    }

    #[tokio::test]
    async fn test_generate_text_api_error_status() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(400)
            .with_body("Bad Request")
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service.generate_text("Hi", None, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_text_malformed_json_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("not json")
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service.generate_text("Hi", None, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_text_empty_candidates() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:generateContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"candidates":[]}"#)
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service.generate_text("Hi", None, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_text_network_error() {
        let settings = settings_with("test-key", "http://127.0.0.1:1");
        let service = GeminiService::new(&settings).unwrap();

        let result = service.generate_text("Hi", None, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_embedding_success() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding":{"values":[0.1,0.2,0.3]}}"#)
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service.get_embedding("some text", None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_get_embedding_with_custom_model() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding":{"values":[1.0]}}"#)
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service
            .get_embedding("text", Some("custom-embed-model"))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_embedding_api_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(500)
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service.get_embedding("text", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_embedding_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("not json")
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let result = service.get_embedding("text", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_embedding_network_error() {
        let settings = settings_with("test-key", "http://127.0.0.1:1");
        let service = GeminiService::new(&settings).unwrap();

        let result = service.get_embedding("text", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_embeddings_multiple_texts() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/v1beta/models/.*:embedContent".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embedding":{"values":[0.5,0.6]}}"#)
            .expect(3)
            .create_async()
            .await;

        let settings = settings_with("test-key", &server.url());
        let service = GeminiService::new(&settings).unwrap();

        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = service.get_embeddings(&texts, None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_get_embeddings_empty_list() {
        let settings = settings_with("test-key", "http://127.0.0.1:1");
        let service = GeminiService::new(&settings).unwrap();

        let texts: Vec<String> = Vec::new();
        let result = service.get_embeddings(&texts, None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_embeddings_propagates_error() {
        let settings = settings_with("test-key", "http://127.0.0.1:1");
        let service = GeminiService::new(&settings).unwrap();

        let texts = vec!["a".to_string()];
        let result = service.get_embeddings(&texts, None).await;
        assert!(result.is_err());
    }
}
