# Provider Utilities (`openai_agent`)

This package contains runtime utilities used by the `openai_agent` workflow.

## What is here

- Provider construction and routing:
  - `model_provider.py` (Azure)
  - `vllm_provider.py` (vLLM)
  - `ollama_provider.py` (Ollama)
  - `unified_provider.py` (provider-agnostic factory)
- Model/deployment helpers:
  - `model_config.py`
  - `azure_models.py`
- Token accounting:
  - `token_tracker.py`
  - `token_tracking.py`

## Configuration inputs

- Azure:
  - `azure_endpoint`
  - `azure_api_version`
  - env fallback: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`
- vLLM:
  - `vllm_endpoint`
  - env fallback: `VLLM_BASE_URL`
- Ollama:
  - `ollama_endpoint`
  - env fallback: `OLLAMA_ENDPOINT`

Local providers fail fast when endpoint configuration is missing.

For command-level examples, refer to the root `README.md`.
