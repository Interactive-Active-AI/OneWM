"""OneWM checkpoint utilities, imported lazily to avoid initialization cycles."""
__all__ = ["infer_backbone_info", "save_checkpoint", "load_checkpoint"]


def __getattr__(name):
    if name in __all__:
        from onewm.utils import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(name)
