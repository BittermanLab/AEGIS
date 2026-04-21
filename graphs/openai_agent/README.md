# OpenAI Agent

Multi-agent workflow with role-specific agents (extractor, event processors, judge) for structured irAE extraction.

## Run

```bash
python main.py openai_agent \
  --model_config default \
  --prompt_variant default \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Local provider example (vLLM):

```bash
python main.py openai_agent \
  --model_config vllm-qwen3-8b-local \
  --provider vllm \
  --vllm-endpoint http://localhost:8001/v1 \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Azure example (AAD/CLI login):

```bash
python main.py openai_agent \
  --model_config default \
  --provider azure \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug \
  --parameters "azure_endpoint=https://<your-endpoint>.openai.azure.com/"
```

## Key Files

- `entry_graph.py`: graph adapter and batch entrypoint
- `processor.py`: provider routing + orchestration
- `parallel_agents/workflow/`: note/event execution flows
- `parallel_agents/agent_factory.py`: role-agent construction
- `model_config.py`: role-to-model mappings

## Notes

- Provider routing supports `azure`, `vllm`, `ollama`, and `auto`.
- For full sweep/evaluation commands, refer to the root `README.md`.