"""Memory adapter — wraps a MemEvolve provider for the Deepagent loop."""

from typing import Any, Callable

from EvolveLab import (
    BaseMemoryProvider,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
)
from EvolveLab.memory_types import TrajectoryData

PROVIDER_MAP: dict[str, str] = {
    "lightweight_memory": "EvolveLab.providers.lightweight_memory_provider:LightweightMemoryProvider",
}

PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "lightweight_memory": {
        "enable_longterm_provision": True,
    },
}


class _ModelWrapper:
    """Wraps a LangChain model as a plain callable for MemEvolve providers."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, messages: list) -> Any:
        return self._model.invoke(messages)


def _import_provider_class(dotted_path: str) -> type:
    module_path, _, class_name = dotted_path.rpartition(":")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class MemoryAdapter:
    """Wraps a BaseMemoryProvider for the Deepagent chat loop.

    Usage:
        adapter = MemoryAdapter("lightweight_memory", model=my_callable)
        adapter.initialize()
        # Before each user query:
        context = adapter.provide_context(user_input)
        # After each agent turn:
        adapter.absorb_trajectory(user_input, messages, result)
    """

    def __init__(
        self,
        provider_name: str = "lightweight_memory",
        model: Callable[..., Any] | None = None,
        storage_dir: str | None = None,
    ):
        if provider_name not in PROVIDER_MAP:
            raise ValueError(
                f"Unknown memory provider {provider_name!r}. "
                f"Available: {list(PROVIDER_MAP)}"
            )
        provider_cls = _import_provider_class(PROVIDER_MAP[provider_name])
        config: dict[str, Any] = dict(PROVIDER_DEFAULTS.get(provider_name, {}))
        if model is not None:
            config["model"] = _ModelWrapper(model)
        if storage_dir is not None:
            config["storage_dir"] = storage_dir
        self._provider: BaseMemoryProvider = provider_cls(config=config)

    def initialize(self) -> bool:
        return self._provider.initialize()

    def provide_context(self, query: str) -> str:
        request = MemoryRequest(
            query=query,
            context="",
            status=MemoryStatus.BEGIN,
        )
        response: MemoryResponse = self._provider.provide_memory(request)
        if not response.memories:
            return ""
        parts = []
        for mem in response.memories:
            content = mem.content if isinstance(mem.content, str) else str(mem.content)
            parts.append(content)
        return "\n\n".join(parts)

    def absorb_trajectory(
        self,
        query: str,
        messages: list,
        result: str | None = None,
        is_success: bool = True,
    ) -> tuple[bool, str]:
        trajectory = []
        for msg in messages:
            role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
            content = getattr(msg, "content", None) or (
                msg if isinstance(msg, str) else str(msg)
            )
            content_str = content if isinstance(content, str) else str(content)
            trajectory.append({"role": role, "content": content_str})
        td = TrajectoryData(
            query=query,
            trajectory=trajectory,
            result=result,
            metadata={"is_correct": is_success, "task_success": is_success},
        )
        return self._provider.take_in_memory(td)

    @property
    def provider(self) -> BaseMemoryProvider:
        return self._provider
