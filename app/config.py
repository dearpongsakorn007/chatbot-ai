from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LINE
    line_channel_secret: str = ""
    line_channel_access_token: str = ""

    # Anthropic
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-5"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Embeddings
    embedding_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"

    # Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""

    # App
    llm_provider: str = "groq"  # "groq" หรือ "claude" สลับได้โดยไม่แก้โค้ด
    top_k: int = 5


settings = Settings()
