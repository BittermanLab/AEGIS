# IRAE Graphs

This repository extracts immune-related adverse events (irAEs) from oncology clinical notes.

It includes three approaches:
- `openai_agent` (multi-agent orchestration)
- `openai_zeroshot` (single-pass prompting)
- `regex` (rule-based baseline)

The pipeline predicts six irAE types (`colitis`, `dermatitis`, `hepatitis`, `myocarditis`, `pneumonitis`, `thyroiditis`) and outputs grade, attribution, and certainty labels.

## Paper/preprint

- [medRxiv preprint (v2)](https://www.medrxiv.org/content/10.64898/2026.02.26.26347179v2)


## Project Layout

```text
config/
  base_config.yaml
  environments/
  sweeps/
data/
  synthetic_notes/
  ctcae.json
graphs/
  openai_agent/
  openai_zeroshot/
  regex/
  agent_split_temporality/
scripts/
  run_sweep.py
  evaluate_sweep.py
  evaluate_all_sweeps.py
main.py
run_sweeps.sh
```

Notes:
- Real-world datasets (`data/rwd/...`) are intentionally not committed to this public repo.
- Outputs are generated locally under `outputs/` unless overridden.

## Setup

### 1) Create environment

```bash
mamba env create -f environment.yml
conda activate irae-graph
```

### 2) Credentials (optional depending on provider)

```bash
export OPENAI_API_KEY="..."
# Azure env vars are optional if using Azure CLI login:
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_KEY="..."
```

## Quick Start (No API Key)

Fastest smoke test (local, no LLM):

```bash
python main.py regex \
  --model_config default \
  --prompt_variant default \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Optional evaluation:

```bash
python scripts/evaluate_sweep.py \
  --sweep-dir outputs/default_sweep \
  --sweep-name default_sweep \
  --output-dir outputs/evaluation_results
```

## Run With LLMs

### Azure OpenAI

If Azure CLI login is configured, this is enough:

```bash
python main.py openai_agent \
  --model_config default \
  --provider azure \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug \
  --parameters "azure_endpoint=https://bwh-bittermanlab-nonprod-openai-service.openai.azure.com/"
```

Notes:
- `azure_endpoint` can be passed via `--parameters` or `AZURE_OPENAI_ENDPOINT`.
- If `AZURE_OPENAI_API_VERSION` is not set, code defaults to `2024-12-01-preview`.
- Azure CLI/AAD token auth is supported (no `OPENAI_API_KEY` required when CLI auth is available).

### Local vLLM / Ollama

When using local model backends, always pass provider and endpoint explicitly:

```bash
python main.py <graph_type> \
  --model_config vllm-qwen3-8b-local \
  --provider vllm \
  --vllm-endpoint http://localhost:8001/v1 \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

Equivalent Ollama flags:

```bash
--provider ollama --ollama-endpoint http://localhost:11434/v1
```

If provider is `vllm` or `ollama` and endpoint is missing, the run fails fast with a clear error.

### Single run

```bash
python main.py openai_agent \
  --model_config 4.1-mini \
  --prompt_variant default \
  --data_dir data/synthetic_notes \
  --output_base_dir outputs \
  --debug
```

### Run a Sweep

```bash
python scripts/run_sweep.py --sweep-file config/sweeps/debug_agent.yaml
```

### Batch Sweeps

```bash
./run_sweeps.sh debug
./run_sweeps.sh dev
./run_sweeps.sh prod --evaluate-all
```

`run_sweeps.sh` expects a conda env named `irae-graph` by default (editable at the top of the script).

## Validation Commands

```bash
pytest -q
python -m compileall .
python main.py openai_agent --help
python main.py openai_zeroshot --help
python main.py agent_split_temporality --help
python main.py regex --help
```

Validated smoke-test matrix:

```bash
python main.py regex --model_config default --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug
python main.py openai_zeroshot --model_config vllm-qwen3-8b-local --provider vllm --vllm-endpoint http://localhost:8001/v1 --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug
python main.py openai_agent --model_config vllm-qwen3-8b-local --provider vllm --vllm-endpoint http://localhost:8001/v1 --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug
python main.py agent_split_temporality --model_config vllm-qwen3-8b-local --provider vllm --vllm-endpoint http://localhost:8001/v1 --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug
```

## Output Structure

Default prediction path:

```text
{output_base_dir}/{sweep_name}/{graph_type}/{model_config}/variant_{prompt_variant}/
  predictions_*.json
  token_usage_*.json
```

Evaluation outputs:

```text
{output_dir}/sweeps/{sweep_name}/
  main/
  appendix/
  visualizations/
```

## Architecture Variants

For agentic ablations:
- `default`: full multi-agent + judge flow
- `ablation_single`: single identifier/grader variant
- `ablation_no_judge`: no-judge variant

See `utils/architecture_diagrams.md` for detailed diagrams.

## Data Hygiene

- No real-world patient dataset is committed in this public clone (`data/rwd` is absent).
- Keep generated predictions and logs out of version control (`outputs/`, `evaluation_results/`, `logs/`, caches).

## Release Checklist

Before publishing an update, run this checklist:

```bash
pytest -q
python -m compileall .
python main.py regex --model_config default --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug
python main.py openai_zeroshot --model_config default --provider azure --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug --parameters "azure_endpoint=https://<your-endpoint>.openai.azure.com/"
python main.py openai_agent --model_config default --provider azure --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug --parameters "azure_endpoint=https://<your-endpoint>.openai.azure.com/"
python main.py agent_split_temporality --model_config default --provider azure --prompt_variant default --data_dir data/synthetic_notes --output_base_dir outputs --debug --parameters "azure_endpoint=https://<your-endpoint>.openai.azure.com/"
```

If all commands pass:
- remove transient artifacts (`outputs/`, `logs/`, caches),
- confirm no real data is present,
- then proceed with the release commit.
