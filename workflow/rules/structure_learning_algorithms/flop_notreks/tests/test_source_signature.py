from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.source_signature_validation import (
    validate_abstract_masks,
    validate_all_dags,
    validate_collider,
)


def test_source_signature_exhaustive_small_dags():
    assert validate_all_dags(5) == 29853
    assert validate_abstract_masks(5) == 33537
    validate_collider()
