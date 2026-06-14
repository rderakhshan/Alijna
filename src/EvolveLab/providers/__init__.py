"""
Memory providers for different frameworks
"""

__all__ = ["AgentKBProvider", "SkillWeaverProvider", "MobileEProvider", "ExpeLProvider"]


def __getattr__(name):
    import importlib
    lazy_map = {
        "AgentKBProvider": ".agent_kb_provider",
        "SkillWeaverProvider": ".skillweaver_provider",
        "MobileEProvider": ".mobilee_provider",
        "ExpeLProvider": ".expel_provider",
    }
    if name in lazy_map:
        module = importlib.import_module(lazy_map[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")