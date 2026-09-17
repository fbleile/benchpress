"""Exhaustive small-DAG validation for source-signature NOTREKS masks."""
from __future__ import annotations

from itertools import product


def _closure(rows: tuple[int, ...], d: int) -> tuple[int, ...]:
    result = [row | (1 << index) for index, row in enumerate(rows)]
    for k in range(d):
        for i in range(d):
            if result[i] & (1 << k):
                result[i] |= result[k]
    return tuple(result)


def _is_dag(rows: tuple[int, ...], d: int) -> bool:
    indegree = [0] * d
    for source in range(d):
        for target in range(d):
            if rows[source] & (1 << target):
                indegree[target] += 1
    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    seen = 0
    while queue:
        source = queue.pop()
        seen += 1
        for target in range(d):
            if rows[source] & (1 << target):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
    return seen == d


def _signatures(rows: tuple[int, ...], d: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    closure = _closure(rows, d)
    sources = tuple(node for node in range(d) if not any(
        rows[source] & (1 << node) for source in range(d)))
    source_bits = {node: 1 << index for index, node in enumerate(sources)}
    signatures = []
    for node in range(d):
        bits = source_bits.get(node, 0)
        for source in sources:
            if closure[source] & (1 << node):
                bits |= source_bits[source]
        signatures.append(bits)
    return sources, tuple(signatures)


def validate_all_dags(max_d: int = 5) -> int:
    checked = 0
    for d in range(1, max_d + 1):
        directed_edges = [(i, j) for i in range(d) for j in range(d) if i != j]
        for edge_mask in range(1 << len(directed_edges)):
            rows = [0] * d
            for bit, (source, target) in enumerate(directed_edges):
                if edge_mask & (1 << bit):
                    rows[source] |= 1 << target
            rows = tuple(rows)
            if not _is_dag(rows, d):
                continue
            _, signatures = _signatures(rows, d)
            for source, target in directed_edges:
                if rows[source] & (1 << target):
                    assert signatures[source] & ~signatures[target] == 0
            closure = _closure(rows, d)
            for left in range(d):
                for right in range(left + 1, d):
                    if not any(
                            closure[source] & (1 << left)
                            and closure[source] & (1 << right)
                            for source in range(d)):
                        assert signatures[left] & signatures[right] == 0
            checked += 1
    return checked


def validate_abstract_masks(max_d: int = 5) -> int:
    """Exhaustively check subset transitivity for every abstract mask size."""
    checked = 0
    for d in range(1, max_d + 1):
        nonempty = range(1, 1 << d)
        for first, second, third in product(nonempty, repeat=3):
            if first & ~second == 0 and second & ~third == 0:
                assert first & ~third == 0
            checked += 1
    return checked


def validate_collider() -> None:
    signatures = (1, 2, 3)  # {1}, {2}, {1,2}
    assert signatures[0] & ~signatures[2] == 0
    assert signatures[1] & ~signatures[2] == 0
    assert signatures[0] & signatures[1] == 0


if __name__ == "__main__":
    dag_count = validate_all_dags()
    abstract_count = validate_abstract_masks()
    validate_collider()
    print(f"validated DAGs: {dag_count}")
    print(f"validated abstract subset triples: {abstract_count}")
    print("validated collider signatures")
