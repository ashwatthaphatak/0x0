__all__ = ["DefensePipeline"]


def __getattr__(name):
    if name == "DefensePipeline":
        from .pipeline import DefensePipeline

        return DefensePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
