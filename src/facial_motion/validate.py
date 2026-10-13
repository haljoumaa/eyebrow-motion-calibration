"""Whole-recording partitions and explicit refusal of duplicated/leaking data."""

from collections import Counter


def assert_disjoint(train, test):
    for key in ["recording_id", "duplicate_group", "sha256"]:
        left = {r[key] for r in train if r.get(key)}
        right = {r[key] for r in test if r.get(key)}
        if left & right:
            raise ValueError(f"Partition leakage via {key}: {left & right}")


def outer_folds(rows, distances=("close", "middle", "far")):
    for held_out in distances:
        train = [r for r in rows if r["distance"] != held_out and r["eyebrow_condition"] == "neutral"]
        test = [r for r in rows if r["distance"] == held_out]
        if not train or not test:
            raise ValueError(f"Unsupported distance fold: {held_out}")
        assert_disjoint(train, test)
        yield held_out, train, test


def inner_folds(rows):
    distances = sorted({r["distance"] for r in rows})
    if len(distances) != 2 or any(r["eyebrow_condition"] != "neutral" for r in rows):
        raise ValueError("Inner validation requires neutral recordings at exactly two training distances")
    for distance in distances:
        train = [r for r in rows if r["distance"] != distance]
        validation = [r for r in rows if r["distance"] == distance]
        assert_disjoint(train, validation)
        yield train, validation


def validate_manifest(rows):
    included = [r for r in rows if str(r["include"]).lower() == "true"]
    if not included:
        raise ValueError("No reviewed recordings included. Complete the data audit before evaluation.")
    for r in included:
        for key in ["condition_verified", "hash_verified"]:
            if str(r[key]).lower() != "true":
                raise ValueError(f"{r['recording_id']}: {key} is not verified")
        if r["head_movement"] != "pitch" or r["distance"] not in {"close", "middle", "far"}:
            raise ValueError("This protocol includes reviewed pitch recordings at the three declared distances only")
        if r["eyebrow_condition"] not in {"neutral", "raised", "alternating"}:
            raise ValueError("Unknown eyebrow condition")
        if r["direction"] not in {"up", "down"}:
            raise ValueError("Verify pitch direction before evaluation")
        if not r["duplicate_group"] or not r["sha256"]:
            raise ValueError("Included recordings need duplicate groups and hashes")
    for key in ["recording_id", "duplicate_group", "sha256"]:
        counts = Counter(r[key] for r in included)
        if any(n > 1 for n in counts.values()):
            raise ValueError(f"Deduplicate included recordings first: {key}")
    for distance in ["close", "middle", "far"]:
        for condition in ["neutral", "raised", "alternating"]:
            if not any(r["distance"] == distance and r["eyebrow_condition"] == condition for r in included):
                raise ValueError(f"Missing reviewed condition: {distance}/{condition}. Revise protocol explicitly if needed.")
    return included
