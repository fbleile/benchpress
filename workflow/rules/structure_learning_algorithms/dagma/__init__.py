from .knowledge import load_sidecar, named_pairs_to_indices, no_trek_pairs_from_dag

__all__ = ["SharedDagmaLinear", "notreks_value_grad", "load_sidecar",
           "named_pairs_to_indices", "no_trek_pairs_from_dag"]


def __getattr__(name):
    """Load the optimizer lazily so oracle sidecars need no dagma wheel."""
    if name in {"SharedDagmaLinear", "notreks_value_grad"}:
        from .shared import SharedDagmaLinear, notreks_value_grad
        return {"SharedDagmaLinear": SharedDagmaLinear,
                "notreks_value_grad": notreks_value_grad}[name]
    raise AttributeError(name)
