# Agent Split Temporality

Variant of the multi-agent workflow that separates past vs current event reasoning before final aggregation.

## Run

```bash
python main.py agent_split_temporality \
  --model_config default \
  --prompt_variant default \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Local provider example (vLLM):

```bash
python main.py agent_split_temporality \
  --model_config vllm-qwen3-8b-local \
  --provider vllm \
  --vllm-endpoint http://localhost:8001/v1 \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Azure example (AAD/CLI login):

```bash
python main.py agent_split_temporality \
  --model_config default \
  --provider azure \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug \
  --parameters "azure_endpoint=https://<your-endpoint>.openai.azure.com/"
```

## Key Files

- `entry_graph.py`: graph adapter entrypoint
- `processor.py`: provider setup + orchestration
- `parallel_agents/workflow/note_processor.py`: temporality split logic
- `parallel_agents/workflow/event_processor.py`: per-event processing/judging
- `model_config.py`: role-to-model mappings

## Notes

- Uses the same output contract as other graph types for common evaluation scripts.
- For sweeps and evaluation commands, use the root `README.md`.