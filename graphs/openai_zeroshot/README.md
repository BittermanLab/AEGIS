# OpenAI Zeroshot

Single-call baseline that extracts irAE labels from each note using one prompt and one model response.

## Run

Preferred entrypoint:

```bash
python main.py openai_zeroshot \
  --model_config default \
  --prompt_variant default \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Local provider example (vLLM):

```bash
python main.py openai_zeroshot \
  --model_config vllm-qwen3-8b-local \
  --provider vllm \
  --vllm-endpoint http://localhost:8001/v1 \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Azure example (AAD/CLI login):

```bash
python main.py openai_zeroshot \
  --model_config default \
  --provider azure \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug \
  --parameters "azure_endpoint=https://<your-endpoint>.openai.azure.com/"
```

## Files

- `entry_graph.py`: framework adapter
- `zeroshot_processor.py`: prompt execution and output normalization
- `model_config.py`: model aliases/config mappings
- `model_provider.py`: provider routing helpers
- `run_processor.py`: direct module runner (optional)

## Notes

- Output format matches other graph types for shared evaluation.
- For project-wide setup, sweep commands, and evaluation, use the root `README.md`.
