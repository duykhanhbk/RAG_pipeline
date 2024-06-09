from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "BAAI/bge-base-en-v1.5" # "BAAI/bge-large-en-v1.5"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512 # 256
    EMBEDDING_SIZE: int = 768
    EMBEDDING_MODEL_DEVICE: str = "cuda"


    ## LLM deployment
    LLMDEPLOY_API_KEY = "YOUR_API_KEY"
    LLMDEPLOY_BASE_URL = "http://0.0.0.0:23333/v1"

    # OpenAI config
    OPENAI_MODEL_ID: str = "gpt-4o" # "gpt-4-1106-preview"
    OPENAI_API_KEY: str | None = None

    # QdrantDB config
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_DATABASE_URL: str = "http://localhost:6333"

    QDRANT_CLOUD_URL: str = "str"
    USE_QDRANT_CLOUD: bool = True
    QDRANT_APIKEY: str | None = None

    # CometML config
    COMET_API_KEY: str
    COMET_WORKSPACE: str
    COMET_PROJECT: str = "llm-rag-deployment"

    # LLM Model config
    TOKENIZERS_PARALLELISM: str = "false"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    MODEL_TYPE: str = "mistralai/Mistral-7B-Instruct-v0.1"
    QWAK_DEPLOYMENT_MODEL_ID: str = "llm_rag_deployment"

    # RAG config
    TOP_K: int = 3
    KEEP_TOP_K: int = 3
    EXPAND_N_QUERY: int = 3


settings = AppSettings()