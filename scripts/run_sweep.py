import yaml  # type: ignore
import os
import subprocess
import logging
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Basic logging configuration - will be overridden by environment settings later
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def configure_logging(logging_config: Dict[str, Any]) -> None:
    """Configure logging based on the provided configuration."""
    if not logging_config:
        return

    # Map string log levels to logging module constants
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Get the log level, defaulting to INFO
    log_level_str = logging_config.get("level", "INFO").upper()
    log_level = log_level_map.get(log_level_str, logging.INFO)

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configurations
        handlers=[logging.StreamHandler()],  # Explicitly use console logging only
    )

    # Set the log level for the current logger
    logger.setLevel(log_level)

    # Set httpx logger to WARNING to suppress HTTP request logs
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Configure openai_agent module loggers
    openai_agent_logger = logging.getLogger("graphs.openai_agent")
    if openai_agent_logger:
        openai_agent_logger.setLevel(log_level)
        # Ensure openai_agent module only logs to console
        openai_agent_logger.handlers = []
        openai_agent_logger.addHandler(logging.StreamHandler())
        openai_agent_logger.propagate = True

    # Set environment variable for token tracking in the openai_agent module
    if log_level == logging.DEBUG:
        os.environ["TOKEN_TRACKER_LOG_LEVEL"] = "DEBUG"
    else:
        os.environ["TOKEN_TRACKER_LOG_LEVEL"] = log_level_str

    logger.info(f"Logging configured with level: {log_level_str}")


@dataclass
class ExperimentConfig:
    graph_type: str
    model_config: str
    environment: Dict[str, Any]
    parameters: Dict[str, Any]
    prompt_variant: Optional[str] = "default"

    def validate_prompt_variant(self, base_config: Dict) -> None:
        """Validate prompt variant configuration"""
        if self.graph_type == "zeroshot":
            variants = base_config["graph_types"]["zeroshot"]["prompt_variants"]
            if self.prompt_variant not in variants:
                raise ValueError(
                    f"Invalid prompt variant '{self.prompt_variant}' for zeroshot. "
                    f"Must be one of: {list(variants.keys())}"
                )
            if not variants[self.prompt_variant].get("enabled", True):
                raise ValueError(f"Prompt variant '{self.prompt_variant}' is disabled")


class ConfigurationManager:
    def __init__(self, base_config_path: str, sweep_config_path: str):
        """
        Initialize configuration manager.

        Args:
            base_config_path: Path to base configuration
            sweep_config_path: Path to sweep configuration file, or name of sweep
        """
        # Load base configuration first
        self.base_config = self._load_config(base_config_path)

        # Check if sweep_config_path is a file or just a name
        if os.path.exists(sweep_config_path):
            # It's a direct file path
            sweep_path = sweep_config_path
        else:
            # It's a sweep name, construct the path
            sweep_path = f"config/sweeps/{sweep_config_path}.yaml"

        # Load sweep details
        self.sweep_details = self._load_config(sweep_path)

        # Save sweep name for later use
        if "name" in self.sweep_details:
            self.sweep_name = self.sweep_details["name"]
        else:
            # Extract name from filename if not specified
            self.sweep_name = os.path.basename(sweep_path).replace(".yaml", "")
            logger.info(
                f"Sweep name not specified in config, using filename: {self.sweep_name}"
            )

        # Load and merge environment config
        self.environment = self._load_environment_config(
            self.sweep_details["environment_config"]
        )

        # Configure logging based on environment settings
        if "logging" in self.environment:
            configure_logging(self.environment["logging"])

    def _load_environment_config(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load environment configuration, handling imports."""
        if "import" in env_config:
            # Load the base environment file
            base_env = self._load_config(env_config["import"])
            # Remove import key and merge configs, with sweep config taking precedence
            env_config.pop("import")

            # Create a nested structure that matches the imported YAML
            merged_env = {
                "paths": {**base_env.get("paths", {}), **env_config.get("paths", {})},
                "runtime": {
                    **base_env.get("runtime", {}),
                    **env_config.get("runtime", {}),
                },
                "logging": {
                    **base_env.get("logging", {}),
                    **env_config.get("logging", {}),
                },
                "parameters": {
                    **base_env.get("parameters", {}),
                    **env_config.get("parameters", {}),
                },
            }

            # Extract runtime settings into top level for easier access
            runtime_settings = merged_env.get("runtime", {})
            merged_env.update(runtime_settings)

            return merged_env
        return env_config

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise ValueError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def validate_configurations(self):
        """Validate that all specified configurations are valid."""
        # Import the GRAPH_TYPE_MAP to check against all valid graph types
        from utils.path_utils import GRAPH_TYPE_MAP

        for experiment in self.sweep_details["experiments"]:
            graph_type = experiment["graph_type"]

            # Check against GRAPH_TYPE_MAP from path_utils instead of base_config
            if graph_type not in GRAPH_TYPE_MAP:
                raise ValueError(
                    f"Invalid graph type: {graph_type}. Available types: {list(GRAPH_TYPE_MAP.keys())}"
                )

            for model in experiment["model_configs"]:
                if model not in self.base_config["model_configs"]:
                    raise ValueError(f"Invalid model config: {model}")

    def get_experiment_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations based on sweep parameters."""
        configs = []
        for experiment in self.sweep_details["experiments"]:
            graph_type = experiment["graph_type"]
            for model_config in experiment["model_configs"]:
                # Create full environment config by merging all settings
                env = self.environment.copy()

                # Add the sweep name to environment
                env["sweep_name"] = self.sweep_name

                # Extract parameters from experiment or use empty dict
                params = experiment.get("parameters", {})

                # Add any global sweep overrides
                if "sweep_overrides" in self.sweep_details:
                    params.update(self.sweep_details["sweep_overrides"])

                config = ExperimentConfig(
                    graph_type=graph_type,
                    model_config=model_config,
                    environment=env,
                    parameters=params,
                    prompt_variant=experiment.get("prompt_variant", "default"),
                )
                config.validate_prompt_variant(self.base_config)
                configs.append(config)
        return configs


class SweepRunner:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run_experiment(self, config: ExperimentConfig):
        """Run a single experiment configuration."""
        logger.info("=" * 80)
        logger.info("Starting experiment with configuration:")
        logger.info("=" * 80)
        logger.info(f"Graph Type:      {config.graph_type}")
        logger.info(f"Model Config:    {config.model_config}")
        logger.info(f"Prompt Variant:  {config.prompt_variant}")

        # Extract settings from environment
        env = config.environment
        debug_mode = env.get("debug", False)
        data_dir = env["paths"]["data_dir"]
        output_base_dir = env["paths"]["output_base_dir"]

        logger.info(f"Debug Mode:      {debug_mode}")
        logger.info(f"Data Directory:  {data_dir}")
        logger.info(
            f"Output Dir:      {output_base_dir}"
        )  # Just show the base output dir

        # Log parameters if any
        if config.parameters:
            logger.info("\nParameters:")
            for key, value in config.parameters.items():
                logger.info(f"  {key}: {value}")

        # Log environment settings
        logger.info("\nEnvironment Settings:")
        for key, value in env.items():
            if key not in ["paths", "parameters"]:  # Skip nested dicts for clarity
                logger.info(f"  {key}: {value}")

        logger.info("=" * 80)
        logger.info("")

        # Construct the command
        cmd = ["python", "main.py"]
        cmd.extend([config.graph_type])
        cmd.extend(["--model_config", config.model_config])
        cmd.extend(["--prompt_variant", config.prompt_variant])
        cmd.extend(["--data_dir", data_dir])
        cmd.extend(["--output_base_dir", output_base_dir])

        # Add debug flag if configured
        if debug_mode:
            cmd.append("--debug")

        # Add parameters as a JSON string, including sweep_name
        params_to_pass = config.parameters.copy() if config.parameters else {}
        # Add sweep_name to parameters so it gets passed to main.py
        params_to_pass["sweep_name"] = env["sweep_name"]

        if params_to_pass:
            import json

            params_json = json.dumps(params_to_pass)
            cmd.extend(["--parameters", params_json])

        # Add environment-specific flags
        if env.get("use_azure", False):
            cmd.append("--use-azure")
            if env.get("azure_endpoint"):
                cmd.extend(["--azure-endpoint", env["azure_endpoint"]])
            if env.get("deployment_name"):
                cmd.extend(["--deployment-name", env["deployment_name"]])

        # Add logging level
        if "logging" in env and "level" in env["logging"]:
            log_level = env["logging"]["level"]
            if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                cmd.extend(["--log-level", log_level])

        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(
                f"Experiment for {config.graph_type} ({config.model_config}) completed successfully!"
            )
            print("=" * 100)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running experiment: {e}")
            return False

    def run_evaluation(self, use_enhanced=False, use_patient_level=False):
        """
        Run the evaluation script to compare all experiments within the sweep.

        Args:
            use_enhanced: Deprecated - always uses enhanced evaluation
            use_patient_level: Whether to run patient-level aggregation before evaluation
        """
        env = self.config_manager.environment
        output_base_dir = env["paths"]["output_base_dir"]
        evaluation_dir = env["paths"]["evaluation_dir"]
        sweep_name = self.config_manager.sweep_name

        # The sweep output directory contains all experiment results
        sweep_output_dir = os.path.join(output_base_dir, sweep_name)

        logger.info(f"Running evaluation for sweep: {sweep_name}")
        logger.info(f"Sweep output directory: {sweep_output_dir}")
        logger.info(f"Results will be saved to: {evaluation_dir}/sweeps/{sweep_name}")

        # Build command for the evaluation script
        cmd = [
            "python",
            "scripts/evaluate_sweep.py",
            "--sweep-dir",
            sweep_output_dir,
            "--sweep-name",
            sweep_name,
            "--output-dir",
            evaluation_dir,
        ]

        try:
            logger.info(f"Running evaluation with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info(f"Evaluation completed successfully for sweep {sweep_name}")
            logger.info(f"")
            logger.info(f"=" * 80)
            logger.info(f"EVALUATION COMPLETE")
            logger.info(f"")
            logger.info(f"Results saved to: {evaluation_dir}/sweeps/{sweep_name}/")
            logger.info(f"  - Main results: main/main_results_table.md")
            logger.info(f"  - Experiment comparisons: appendix/appendix_table3a-c.md")
            logger.info(f"  - Processing summary: appendix/appendix_table3d.md")
            logger.info(f"  - Visualizations: visualizations/")
            logger.info(f"=" * 80)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running evaluation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during evaluation: {e}")
            raise

        return

    def run_sweeps(
        self, skip_eval=False, force_enhanced_eval=False, use_patient_level=False
    ):
        """
        Run all configured sweeps.

        Args:
            skip_eval: Whether to skip the evaluation phase
            force_enhanced_eval: Whether to force using the enhanced evaluation pipeline
            use_patient_level: Whether to run patient-level aggregation before evaluation
        """
        configs = self.config_manager.get_experiment_configs()
        logger.info(f"Running {len(configs)} experiments")

        # Check if we should run experiments sequentially
        run_sequential = False
        for config in configs:
            if config.graph_type in ["openai_agent", "agent_split_temporality"]:
                run_sequential = True
                logger.info(
                    "OpenAI Agent workflows detected. Running experiments sequentially to avoid asyncio issues."
                )
                break

        # Run experiments
        if run_sequential:
            # Run experiments one by one
            for config in configs:
                self.run_experiment(config)
        else:
            # Run experiments in parallel using ThreadPoolExecutor
            max_workers = min(4, len(configs))  # Limit parallel experiments
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(self.run_experiment, configs))

        # Run evaluation unless skipped
        if not skip_eval:
            logger.info("\nStarting evaluation phase...")
            # Force enhanced evaluation for all sweeps
            self.run_evaluation(
                use_enhanced=True,
                use_patient_level=use_patient_level,
            )
        else:
            logger.info("\nSkipping evaluation phase as requested.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment sweeps")
    parser.add_argument(
        "--base-config",
        default="config/base_config.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--sweep",
        help="Name of the sweep configuration (without .yaml extension)",
    )
    parser.add_argument(
        "--sweep-file",
        help="Direct path to sweep configuration file",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the evaluation phase after running experiments",
    )
    parser.add_argument(
        "--enhanced-eval",
        action="store_true",
        help="Force using the enhanced evaluation pipeline (always True now)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing outputs, skip experiments",
    )
    parser.add_argument(
        "--patient-level-eval",
        action="store_true",
        help="Run patient-level aggregation before evaluation",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override logging level from configuration",
    )

    args = parser.parse_args()

    # Determine sweep config path
    if args.sweep_file:
        sweep_config_path = args.sweep_file
    elif args.sweep:
        sweep_config_path = args.sweep
    else:
        parser.error("Either --sweep or --sweep-file must be specified")

    try:
        # Load configurations
        config_manager = ConfigurationManager(args.base_config, sweep_config_path)

        # Override log level if specified
        if args.log_level:
            logging_config = {"level": args.log_level}
            configure_logging(logging_config)

        # Validate configurations
        config_manager.validate_configurations()

        # Initialize runner
        runner = SweepRunner(config_manager)

        # Get configurations and print them
        configs = config_manager.get_experiment_configs()

        print("====== Running with the following configurations ======\n")
        for i, config in enumerate(configs, 1):
            print(f"Configuration {i}:")
            print(f"  Graph Type: {config.graph_type}")
            print(f"  Model Config: {config.model_config}")
            print(f"  Environment: {config.environment}")
            print(f"  Prompt Variant: {config.prompt_variant}")
            print(f"  Parameters: {config.parameters}")
            print()
        print("=====================================================\n")

        # Run sweeps or just evaluation
        if args.eval_only:
            logger.info("Running evaluation only on existing outputs...")
            runner.run_evaluation(
                use_enhanced=True,
                use_patient_level=args.patient_level_eval,
            )
        else:
            runner.run_sweeps(
                skip_eval=args.skip_eval,
                force_enhanced_eval=True,  # Always use enhanced evaluation
                use_patient_level=args.patient_level_eval,
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
