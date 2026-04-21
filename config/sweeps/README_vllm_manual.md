# Manual vLLM Serving Workflow

## Overview
Three vLLM sweep configurations are available for different testing scenarios:
- `vllm_debug.yaml` - Quick testing with 5 samples
- `vllm_dev.yaml` - Development testing with 20 samples  
- `vllm_prod.yaml` - Full production run with entire dataset

All require manual vLLM server management where you start the server and comment/uncomment the appropriate model in the sweep config.

## Workflow

### 1. Start vLLM Server
First, manually start the vLLM server for the model you want to test:

```bash
# Example commands for different models:

# Qwen3-32B (requires XFORMERS backend)
export VLLM_ATTENTION_BACKEND=XFORMERS
nohup vllm serve Qwen/Qwen3-32B-FP8 --chat-template ./utils/qwen3_nonthinking.jinja --tensor-parallel-size 2 > vllm_server.log 2>&1 &

# DeepSeek R1 8B (requires XFORMERS backend)
export VLLM_ATTENTION_BACKEND=XFORMERS
nohup vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --tensor-parallel-size 2 > vllm_server.log 2>&1 &

# DeepSeek R1 Distill Qwen 32B (requires XFORMERS backend)
export VLLM_ATTENTION_BACKEND=XFORMERS
nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager > vllm_server.log 2>&1 &

# Gemma3-27B (requires FLASHINFER backend)
export VLLM_ATTENTION_BACKEND=FLASHINFER
nohup vllm serve google/gemma-3-27b-it --tensor-parallel-size 2 --enable-chunked-prefill > vllm_server.log 2>&1 &

# Other Qwen models
export VLLM_ATTENTION_BACKEND=XFORMERS
nohup vllm serve Qwen/QwQ-32B-Preview --chat-template ./utils/qwen3_nonthinking.jinja --tensor-parallel-size 2 > vllm_server.log 2>&1 &
nohup vllm serve Qwen/Qwen3-8B-FP8 --chat-template ./utils/qwen3_nonthinking.jinja > vllm_server.log 2>&1 &
nohup vllm serve Qwen/Qwen3-14B-FP8 --chat-template ./utils/qwen3_nonthinking.jinja > vllm_server.log 2>&1 &

# Gemma models
export VLLM_ATTENTION_BACKEND=FLASHINFER
nohup vllm serve google/gemma-3-4b-it > vllm_server.log 2>&1 &
nohup vllm serve google/gemma-3-12b-it > vllm_server.log 2>&1 &
```

### 2. Edit the Sweep Config
Choose your sweep config based on testing needs:
- `vllm_debug.yaml` - 5 samples, batch size 2, DEBUG logging
- `vllm_dev.yaml` - 20 samples, batch size 5, INFO logging
- `vllm_prod.yaml` - Full dataset, batch size 10, gold error eval enabled

Open the chosen config and:
- Comment out all models except the one matching your running server
- For example, if running `vllm-qwen3-32b`, ensure only that line is uncommented:
```yaml
model_configs: 
  - vllm-qwen3-32b  # Active
  # - vllm-gemma3-27b  # Commented out
  # - vllm-deepseek-r1-8b  # Commented out
```

### 3. Run the Sweep
```bash
# For quick debugging (5 samples)
python scripts/run_sweep.py --sweep-file config/sweeps/vllm_debug.yaml

# For development testing (20 samples)
python scripts/run_sweep.py --sweep-file config/sweeps/vllm_dev.yaml

# For production runs (full dataset)
python scripts/run_sweep.py --sweep-file config/sweeps/vllm_prod.yaml
```

### 4. Repeat for Other Models
- Stop the current vLLM server (Ctrl+C)
- Start a new vLLM server for the next model
- Update the sweep config to uncomment the new model
- Run the sweep again

## Important Notes

1. **One Model at a Time**: Only uncomment ONE model in each experiment section to match your running server

2. **Server Must Match Config**: The model alias in the config (e.g., `vllm-qwen3-32b`) must match the model you're serving

3. **Check Endpoint**: Ensure the sweep/provider endpoint matches the running vLLM server (for example, `http://localhost:8001/v1`).

4. **Attention Backend**: Some models require specific attention backends:
   - Qwen models: Use `VLLM_ATTENTION_BACKEND=XFORMERS`
   - Gemma models: Use `VLLM_ATTENTION_BACKEND=FLASHINFER`
   - DeepSeek models: Use `VLLM_ATTENTION_BACKEND=XFORMERS`

5. **Config Differences**:
   - **Debug**: 5 samples, batch_size=2, DEBUG logging
   - **Dev**: 20 samples, batch_size=5, INFO logging  
   - **Prod**: Full dataset, batch_size=10, gold error eval, higher concurrency
