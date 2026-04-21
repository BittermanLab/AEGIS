#!/bin/bash

# --- Configuration ---
# Name of the conda environment to use
CONDA_ENV_NAME="irae-graph"
# CONDA_ENV_NAME="openai_agent_env"
# CONDA_ENV_NAME="openai-agent"

# Adjust the path to your main sweep execution script if needed
SWEEP_SCRIPT="scripts/run_sweep.py"
# Directory containing the sweep configurations
CONFIG_DIR="config/sweeps"
# --- End Configuration ---

# Function to print usage instructions
usage() {
  echo "Usage: $0 <environment> [--evaluate-all]"
  echo "       $0 vllm_<environment> [--evaluate-all]"
  echo "       $0 snowflake_<environment> [--evaluate-all]"
  echo "  environment: One of 'debug', 'dev', or 'prod'"
  echo "  --evaluate-all: Optional flag to run cross-sweep evaluation after sweeps complete"
  echo ""
  echo "Examples:"
  echo "  $0 debug              # Run debug agent sweeps only"
  echo "  $0 vllm_debug         # Run vllm debug sweeps only"
  echo "  $0 snowflake_dev      # Run snowflake dev sweeps only"
  echo "  $0 prod --evaluate-all # Run prod sweeps and generate final paper results"
  exit 1
}

# --- Conda Setup ---
# Try to activate the conda environment
echo "Attempting to activate conda environment: $CONDA_ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh" || {
  echo "Error: Failed to source conda profile. Make sure conda is initialized."
  exit 1
}
conda activate "$CONDA_ENV_NAME" || {
  echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'."
  echo "Please ensure the environment exists and is properly configured."
  exit 1
}
echo "Conda environment '$CONDA_ENV_NAME' activated successfully."
echo "Using Python: $(which python)"

# --- Argument Parsing ---
# Check if an environment argument is provided
if [ -z "$1" ]; then
  echo "Error: Missing environment argument."
  usage
fi

# Validate the environment argument
ENVIRONMENT="$1"

# Check if it's a vllm environment
if [[ "$ENVIRONMENT" == vllm_* ]]; then
  # Extract the base environment from vllm_<env>
  BASE_ENV="${ENVIRONMENT#vllm_}"
  if [[ "$BASE_ENV" != "debug" && "$BASE_ENV" != "dev" && "$BASE_ENV" != "prod" ]]; then
    echo "Error: Invalid vllm environment '$ENVIRONMENT'. Must be vllm_debug, vllm_dev, or vllm_prod."
    usage
  fi
  IS_VLLM=true
  IS_SNOWFLAKE=false
# Check if it's a snowflake environment
elif [[ "$ENVIRONMENT" == snowflake_* ]]; then
  # Extract the base environment from snowflake_<env>
  BASE_ENV="${ENVIRONMENT#snowflake_}"
  if [[ "$BASE_ENV" != "debug" && "$BASE_ENV" != "dev" && "$BASE_ENV" != "prod" ]]; then
    echo "Error: Invalid snowflake environment '$ENVIRONMENT'. Must be snowflake_debug, snowflake_dev, or snowflake_prod."
    usage
  fi
  IS_VLLM=false
  IS_SNOWFLAKE=true
else
  # Regular environment
  if [[ "$ENVIRONMENT" != "debug" && "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
    echo "Error: Invalid environment '$ENVIRONMENT'."
    usage
  fi
  IS_VLLM=false
  IS_SNOWFLAKE=false
  BASE_ENV="$ENVIRONMENT"
fi

# --- Sweep Execution ---
# Construct the pattern based on the environment and vllm/snowflake flag
if [ "$IS_VLLM" = true ]; then
  SWEEP_PATTERN="vllm_${BASE_ENV}.yaml"
elif [ "$IS_SNOWFLAKE" = true ]; then
  SWEEP_PATTERN="snowflake_${BASE_ENV}_agent.yaml"
else
  SWEEP_PATTERN="${BASE_ENV}_agent.yaml"
fi

# Exit immediately if a command exits with a non-zero status.
set -e

echo "\nStarting '$ENVIRONMENT' sweep runs..."
echo "Looking for configurations matching '$SWEEP_PATTERN' in '$CONFIG_DIR'..."

# Create an array to store the names of all sweeps that were run
EXECUTED_SWEEPS=()

found_sweeps=0
# Loop through all files matching the pattern in the specified directory
for config_file in "$CONFIG_DIR"/$SWEEP_PATTERN; do
  # Check if the matched item is actually a file (handles cases where the pattern matches nothing)
  if [ -f "$config_file" ]; then
    found_sweeps=$((found_sweeps + 1))
    echo "-----------------------------------------------------"
    echo "Running sweep with config: $config_file"
    echo "-----------------------------------------------------"

    # Extract the sweep name from the config file for later use
    SWEEP_NAME=$(basename "$config_file" .yaml)
    EXECUTED_SWEEPS+=("$SWEEP_NAME")

    # Execute the sweep script using the python from the activated conda environment
    python "$SWEEP_SCRIPT" --sweep-file "$config_file"

    echo "-----------------------------------------------------"
    echo "Finished sweep: $config_file"
    echo "-----------------------------------------------------"

    echo # Add a blank line for readability
  fi
done

if [ "$found_sweeps" -eq 0 ]; then
  echo "Warning: No sweep configurations found matching '$SWEEP_PATTERN' in '$CONFIG_DIR'."
else
  echo "-----------------------------------------------------"
  echo "All $found_sweeps '$ENVIRONMENT' sweeps completed successfully!"
  echo "-----------------------------------------------------"
  echo "Sweeps executed:"
  for sweep in "${EXECUTED_SWEEPS[@]}"; do
    echo "  - $sweep"
  done
  echo "-----------------------------------------------------"
fi

# Optionally run cross-sweep evaluation if requested
if [ "$2" == "--evaluate-all" ]; then
  echo "-----------------------------------------------------"
  echo "Running cross-sweep evaluation..."
  echo "-----------------------------------------------------"
  
  # Determine output directory based on environment
  if [ "$BASE_ENV" == "prod" ]; then
    OUTPUT_DIR="data/rwd/prod_outputs"
  elif [ "$BASE_ENV" == "dev" ]; then
    OUTPUT_DIR="data/rwd/dev_outputs"
  else
    OUTPUT_DIR="data/synthetic_outputs"
  fi
  
  python scripts/evaluate_all_sweeps.py --output-base-dir "$OUTPUT_DIR" --results-dir "results"
  
  if [ $? -eq 0 ]; then
    echo "-----------------------------------------------------"
    echo "Cross-sweep evaluation completed successfully!"
    echo "Final results saved to: results/"
    echo "-----------------------------------------------------"
  else
    echo "Error: Cross-sweep evaluation failed"
  fi
fi

# Deactivate conda environment upon exit (optional, but good practice)
trap "conda deactivate" EXIT
