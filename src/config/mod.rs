use crate::error::{AppError, AppResult};
use once_cell::sync::Lazy;

pub static SETTINGS: Lazy<Settings> = Lazy::new(|| settings_from_load_result(Settings::load()));

/// Falls back to defaults if loading failed, logging why. Split out from the
/// `Lazy` initializer so the fallback branch can be tested directly: real
/// `Settings::load()` calls never currently fail, so without this the branch
/// would be dead code from a test's point of view.
fn settings_from_load_result(result: AppResult<Settings>) -> Settings {
    result.unwrap_or_else(|e| {
        eprintln!("Failed to load settings: {}", e);
        Settings::default()
    })
}

pub fn get_settings() -> &'static Settings {
    &SETTINGS
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub app_name: String,
    pub version: String,
    pub description: String,
    pub server_host: String,
    pub server_port: u16,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub gemini_embedding_model: String,
    pub gemini_api_base_url: String,
    pub qdrant_url: String,
    pub qdrant_api_key: Option<String>,
    pub collection_name: String,
    pub default_chunk_size: usize,
    pub default_chunk_overlap: usize,
    pub cors_origins: Vec<String>,
    pub environment: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            app_name: "Gemini Bridge".to_string(),
            version: "0.1.0".to_string(),
            description: "AI / LLM Gemini Bridge & RAG Service".to_string(),
            server_host: "127.0.0.1".to_string(),
            server_port: 8000,
            gemini_api_key: String::new(),
            gemini_model: "gemini-3.8-flash".to_string(),
            gemini_embedding_model: "gemini-embedding-2".to_string(),
            gemini_api_base_url: "https://generativelanguage.googleapis.com".to_string(),
            qdrant_url: "http://localhost:6333".to_string(),
            qdrant_api_key: None,
            collection_name: "rag_knowledge_base".to_string(),
            default_chunk_size: 800,
            default_chunk_overlap: 100,
            cors_origins: vec!["*".to_string()],
            environment: "development".to_string(),
        }
    }
}

impl Settings {
    pub fn load() -> AppResult<Self> {
        dotenv::dotenv().ok();
        let mut settings = Settings::default();

        if let Ok(val) = std::env::var("APP_NAME") {
            settings.app_name = val;
        }
        if let Ok(val) = std::env::var("SERVER_HOST") {
            settings.server_host = val;
        }
        if let Ok(val) = std::env::var("SERVER_PORT") {
            settings.server_port = val.parse().unwrap_or(8000);
        }

        settings.gemini_api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .unwrap_or_default();

        if let Ok(val) = std::env::var("GEMINI_MODEL") {
            settings.gemini_model = val;
        }
        if let Ok(val) = std::env::var("GEMINI_EMBEDDING_MODEL") {
            settings.gemini_embedding_model = val;
        }
        if let Ok(val) = std::env::var("GEMINI_API_BASE_URL") {
            settings.gemini_api_base_url = val;
        }

        if let Ok(val) = std::env::var("QDRANT_URL") {
            settings.qdrant_url = val;
        }
        if let Ok(val) = std::env::var("QDRANT_API_KEY") {
            settings.qdrant_api_key = Some(val);
        }

        if settings.gemini_api_key.is_empty() {
            eprintln!("Warning: GEMINI_API_KEY is not configured");
        }

        Ok(settings)
    }

    pub fn is_production(&self) -> bool {
        self.environment.to_lowercase() == "production"
    }

    pub fn validate(&self) -> AppResult<()> {
        if self.gemini_api_key.is_empty() {
            return Err(AppError::ConfigError(
                "GEMINI_API_KEY is not configured".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serializes tests that mutate process-wide environment variables so
    // they can't interleave with each other within this test binary.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.app_name, "Gemini Bridge");
        assert_eq!(settings.version, "0.1.0");
        assert_eq!(settings.server_host, "127.0.0.1");
        assert_eq!(settings.server_port, 8000);
        assert_eq!(settings.gemini_model, "gemini-3.8-flash");
        assert_eq!(settings.gemini_embedding_model, "gemini-embedding-2");
        assert_eq!(
            settings.gemini_api_base_url,
            "https://generativelanguage.googleapis.com"
        );
        assert_eq!(settings.qdrant_url, "http://localhost:6333");
        assert!(settings.qdrant_api_key.is_none());
        assert_eq!(settings.collection_name, "rag_knowledge_base");
        assert_eq!(settings.default_chunk_size, 800);
        assert_eq!(settings.default_chunk_overlap, 100);
        assert_eq!(settings.cors_origins, vec!["*".to_string()]);
        assert_eq!(settings.environment, "development");
    }

    #[test]
    fn test_settings_validation_without_key() {
        let mut settings = Settings::default();
        settings.gemini_api_key = String::new();
        assert!(settings.validate().is_err());
    }

    #[test]
    fn test_settings_validation_with_key() {
        let mut settings = Settings::default();
        settings.gemini_api_key = "test-key".to_string();
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_is_production_lowercase() {
        let mut settings = Settings::default();
        settings.environment = "development".to_string();
        assert!(!settings.is_production());

        settings.environment = "production".to_string();
        assert!(settings.is_production());
    }

    #[test]
    fn test_is_production_case_insensitive() {
        let mut settings = Settings::default();
        settings.environment = "PRODUCTION".to_string();
        assert!(settings.is_production());

        settings.environment = "Staging".to_string();
        assert!(!settings.is_production());
    }

    #[test]
    fn test_settings_clone_and_debug() {
        let settings = Settings::default();
        let cloned = settings.clone();
        assert_eq!(cloned.app_name, settings.app_name);
        let debug_str = format!("{:?}", settings);
        assert!(debug_str.contains("Gemini Bridge"));
    }

    #[test]
    fn test_get_settings_returns_static_ref() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let s1 = get_settings();
        let s2 = get_settings();
        assert_eq!(s1.app_name, s2.app_name);
        assert_eq!(s1 as *const Settings, s2 as *const Settings);
    }

    #[test]
    fn test_load_reads_env_overrides() {
        let _guard = ENV_MUTEX.lock().unwrap();

        std::env::set_var("APP_NAME", "Test App");
        std::env::set_var("SERVER_HOST", "0.0.0.0");
        std::env::set_var("SERVER_PORT", "9999");
        std::env::set_var("GEMINI_API_KEY", "env-test-key");
        std::env::set_var("GEMINI_MODEL", "test-model");
        std::env::set_var("GEMINI_EMBEDDING_MODEL", "test-embedding");
        std::env::set_var("GEMINI_API_BASE_URL", "http://localhost:9000");
        std::env::set_var("QDRANT_URL", "http://localhost:1234");
        std::env::set_var("QDRANT_API_KEY", "qdrant-secret");

        let settings = Settings::load().expect("load should succeed");

        assert_eq!(settings.app_name, "Test App");
        assert_eq!(settings.server_host, "0.0.0.0");
        assert_eq!(settings.server_port, 9999);
        assert_eq!(settings.gemini_api_key, "env-test-key");
        assert_eq!(settings.gemini_model, "test-model");
        assert_eq!(settings.gemini_embedding_model, "test-embedding");
        assert_eq!(settings.gemini_api_base_url, "http://localhost:9000");
        assert_eq!(settings.qdrant_url, "http://localhost:1234");
        assert_eq!(settings.qdrant_api_key, Some("qdrant-secret".to_string()));

        std::env::remove_var("APP_NAME");
        std::env::remove_var("SERVER_HOST");
        std::env::remove_var("SERVER_PORT");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("GEMINI_MODEL");
        std::env::remove_var("GEMINI_EMBEDDING_MODEL");
        std::env::remove_var("GEMINI_API_BASE_URL");
        std::env::remove_var("QDRANT_URL");
        std::env::remove_var("QDRANT_API_KEY");
    }

    #[test]
    fn test_load_falls_back_to_google_api_key() {
        let _guard = ENV_MUTEX.lock().unwrap();

        std::env::remove_var("GEMINI_API_KEY");
        std::env::set_var("GOOGLE_API_KEY", "google-fallback-key");

        let settings = Settings::load().expect("load should succeed");
        assert_eq!(settings.gemini_api_key, "google-fallback-key");

        std::env::remove_var("GOOGLE_API_KEY");
    }

    #[test]
    fn test_load_invalid_port_falls_back_to_default() {
        let _guard = ENV_MUTEX.lock().unwrap();

        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::set_var("SERVER_PORT", "not-a-number");

        let settings = Settings::load().expect("load should succeed");
        assert_eq!(settings.server_port, 8000);

        std::env::remove_var("SERVER_PORT");
    }

    #[test]
    fn test_settings_from_load_result_ok_passes_through() {
        let settings = settings_from_load_result(Ok(Settings::default()));
        assert_eq!(settings.app_name, "Gemini Bridge");
    }

    #[test]
    fn test_settings_from_load_result_err_falls_back_to_default() {
        let settings =
            settings_from_load_result(Err(AppError::ConfigError("boom".to_string())));
        assert_eq!(settings.app_name, Settings::default().app_name);
        assert_eq!(settings.server_port, Settings::default().server_port);
    }

    #[test]
    fn test_load_with_no_api_key_still_succeeds() {
        let _guard = ENV_MUTEX.lock().unwrap();

        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("GOOGLE_API_KEY");

        let settings = Settings::load().expect("load should succeed even without a key");
        assert!(settings.gemini_api_key.is_empty());
    }
}
