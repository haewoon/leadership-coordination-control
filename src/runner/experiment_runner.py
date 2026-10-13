from __future__ import annotations

from pathlib import Path
from typing import Any

from src.logging.run_logger import RunLogger
from src.models.openai_compat_model import OpenAICompatModel
from src.models.mock_model import MockModel
from src.models.ollama_model import OllamaModel
from src.models.openai_model import OpenAIModel
from src.policies import POLICY_REGISTRY
from src.runner.episode_runner import EpisodeRunner
from src.tasks import TASK_REGISTRY
from src.utils.seed import set_global_seed


class ExperimentRunner:
    def __init__(self, config: dict[str, Any], root_dir: Path) -> None:
        self.config = config
        self.root_dir = root_dir
        self.output_dir = root_dir / config["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_logger = RunLogger(self.output_dir)

    def _build_model(self):
        provider = self.config["model"]["provider"]
        if provider == "mock":
            return MockModel(name=self.config["model"]["name"])
        if provider == "ollama":
            return OllamaModel(
                name=self.config["model"]["name"],
                host=self.config["model"].get("host", "http://localhost:11434"),
                generation=self.config.get("generation", {}),
            )
        if provider == "openai":
            return OpenAIModel(
                name=self.config["model"]["name"],
                base_url=self.config["model"].get("base_url", "https://api.openai.com/v1"),
                generation=self.config.get("generation", {}),
                api_key_env=self.config["model"].get("api_key_env", "OPENAI_API_KEY"),
            )
        if provider == "openai_compat":
            return OpenAICompatModel(
                name=self.config["model"]["name"],
                base_url=self.config["model"].get("base_url"),
                generation=self.config.get("generation", {}),
                api_key_env=self.config["model"].get("api_key_env", "LLM_API_KEY"),
            )
        raise ValueError(f"Unsupported model provider: {provider}")

    def run(self) -> None:
        model = self._build_model()
        warmup_prompt = self.config.get("warmup_prompt")
        if warmup_prompt:
            model.generate(prompt=warmup_prompt, metadata={"phase": "warmup"})
        episode_runner = EpisodeRunner(
            config=self.config,
            model=model,
            run_logger=self.run_logger,
        )

        for seed in self.config["seeds"]:
            set_global_seed(seed)
            for task_cfg in self.config["tasks"]:
                task_name = task_cfg["name"]
                task_loader = TASK_REGISTRY[task_name]
                task_instances = task_loader(self.root_dir / task_cfg["path"], task_cfg.get("limit"))
                for condition in self.config["conditions"]:
                    policy_cls = POLICY_REGISTRY[condition]
                    policy = policy_cls(self.config)
                    if getattr(policy_cls, "needs_runtime", False):
                        policy.model = model
                        policy.run_logger = self.run_logger
                    for instance in task_instances:
                        episode_runner.run_episode(
                            task_instance=instance,
                            policy=policy,
                            seed=seed,
                            condition=condition,
                        )
