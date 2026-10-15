"""OneWM: unified causal world and action modeling."""

__version__ = "0.1.0"
__all__ = ["build_framework", "build_trainer", "WAMModel", "MoTWAM"]


def __getattr__(name):
    if name in {"build_framework", "build_trainer"}:
        from onewm import builder

        return getattr(builder, name)
    if name in {"WAMModel", "MoTWAM"}:
        from onewm import wam

        return getattr(wam, name)
    raise AttributeError(name)
