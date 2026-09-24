from configuration import DigitalConfig, utils

class Configuration(DigitalConfig):
    llm_model_name: str = 'Qwen3-32B'
    llm_base_url: str = 'http://dtl-gpu26:8000/v1'
    llm_api_key: str = 'smth'
    llm_system_prompt: str = 'You are a helpful assistant.'
    llm_parse_retray: int = 3
    # Embedder and reranker are served by Infinity; model names must match its model ids.
    embedder_name: str = 'models/USER-bge-m3'
    emb_base_url: str = 'http://infinity:7997'
    emb_api_key: str = ''
    emb_size: int = 768
    reranker_name: str = 'models/bge-reranker-v2-m3'
    reranker_base_url: str = 'http://infinity:7997'
    reranker_api_key: str = ''
    use_reranker: bool = True
    k_document: int = 10
    qdrant_url: str = 'https://qdrant.k8s-test'
    qdrant_collection: str = 'documents'
    max_tasks: int = 10
    metrics_topic: str = 'nlp_assistant_metrics'
    backend_callback_url: str = 'https://nlp-assistant-backend.k8s/api/v1/model_result'

    @property
    def app_name(self) -> str:
        return utils.get_app_name()
