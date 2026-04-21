from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List


DEPLOYMENT_COSTS = {
    "o1": {
        "prompt": 15.0,
        "completion": 60.0,
    },
    "gpt-4o-mini-jg": {
        "prompt": 0.15,
        "completion": 0.60,
    },
    "o3-mini-jg": {
        "prompt": 1.10,
        "completion": 4.40,
    },
    "o4-mini": {
        "prompt": 1.10,
        "completion": 4.40,
    },
}


def get_cost_for_deployment(deployment_name: str) -> dict:
    dep_lower = deployment_name.lower()
    for known_key, cost_info in DEPLOYMENT_COSTS.items():
        if known_key in dep_lower:
            return cost_info
    # fallback
    return {"prompt": 0.0, "completion": 0.0}


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Tracks token usage by model name and computes cost per model."""

    def __init__(self, deployment_name: str = None):
        super().__init__()
        # usage_by_model: { model_name: {prompt_tokens, completion_tokens, total_tokens, cost info...} }
        self.usage_by_model: Dict[str, Dict[str, float]] = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.deployment_name = deployment_name
        # We'll store "current_model" at run time

    def on_llm_start(
        self, serialized: Dict[str, Any], messages: List[Any], **kwargs: Any
    ) -> None:
        """Called at the start of an LLM run."""
        init_args = serialized.get("init_args", {})
        # Use provided deployment name if available, otherwise extract from serialized
        if self.deployment_name:
            model_name = self.deployment_name
        else:
            model_name = (
                init_args.get("azure_deployment")
                or init_args.get("model_name")
                or serialized.get("name", "unknown")
            )

        # Standardize or rename if you prefer:
        if "gpt-4o-mini" in model_name.lower():
            model_name = "gpt-4o-mini-jg"
        elif "o3-mini" in model_name.lower():
            model_name = "o3-mini-jg"
        elif "o1" in model_name.lower():
            model_name = "o1"

        # Store on self so we can update usage in on_llm_end
        self.current_model = model_name

        # If this model hasn't been tracked yet, create an entry
        if self.current_model not in self.usage_by_model:
            cost_info = get_cost_for_deployment(self.current_model)
            self.usage_by_model[self.current_model] = {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "prompt_cost_per_million": cost_info.get("prompt", 0.0),
                "completion_cost_per_million": cost_info.get("completion", 0.0),
            }

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Called at the end of an LLM run. We read token usage from response.llm_output["token_usage"].
        """
        if response.llm_output is None:
            return

        usage = response.llm_output.get("token_usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = prompt + completion

        if hasattr(self, "current_model") and self.current_model in self.usage_by_model:
            self.usage_by_model[self.current_model]["prompt_tokens"] += prompt
            self.usage_by_model[self.current_model]["completion_tokens"] += completion
            self.usage_by_model[self.current_model]["total_tokens"] += total

        # Update overall totals
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total

    def get_model_costs(self) -> Dict[str, Dict[str, float]]:
        """
        Return usage & cost for each model.
        We compute cost by using the model's stored cost rates in usage_by_model.
        """
        costs = {}
        for model_name, usage in self.usage_by_model.items():
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
            prompt_rate = usage["prompt_cost_per_million"]
            completion_rate = usage["completion_cost_per_million"]

            prompt_cost = (prompt_tokens / 1_000_000.0) * prompt_rate
            completion_cost = (completion_tokens / 1_000_000.0) * completion_rate
            total_cost = prompt_cost + completion_cost

            costs[model_name] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": total_cost,
            }
        return costs

    def get_overall_costs(self) -> Dict[str, float]:
        """
        Aggregate cost from all models.
        """
        model_costs = self.get_model_costs()
        prompt_cost = sum(m["prompt_cost"] for m in model_costs.values())
        completion_cost = sum(m["completion_cost"] for m in model_costs.values())
        total_cost = prompt_cost + completion_cost

        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        }

    def reset_usage(self) -> None:
        self.usage_by_model.clear()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        if hasattr(self, "current_model"):
            del self.current_model
