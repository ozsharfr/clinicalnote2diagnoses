"""Phase 0 — data plumbing.

Load visits + annotations, build the majority-vote gold (D2), pick the kept
closed label set (D1), and split by patient so no patient leaks across the
train/test boundary (E2).

    annotations ──► gold[visit] = {codes with >=2 annotators}
                └─► kept_codes  = codes with >= MIN_CODE_MENTIONS mentions
    visits.patient_id ──► train / test visit ids (grouped by patient)
"""
from collections import defaultdict
import numpy as np
import pandas as pd

from . import config


def load_visits() -> pd.DataFrame:
    df = pd.read_csv(config.VISITS_CSV, dtype=str).fillna("")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def load_annotations() -> pd.DataFrame:
    return pd.read_csv(config.ANNOTATIONS_CSV, dtype=str).fillna("")


def build_gold(ann: pd.DataFrame, min_annotators: int = config.MIN_ANNOTATORS_FOR_GOLD) -> dict:
    """visit_id -> set of codes agreed by >= min_annotators distinct annotators."""
    votes = defaultdict(lambda: defaultdict(set))  # visit -> code -> {annotators}
    for v, code, a in zip(ann["visit_id"], ann["icd10cm_code"], ann["annotator_id"]):
        votes[v][code].add(a)
    return {
        v: {c for c, anns in cm.items() if len(anns) >= min_annotators}
        for v, cm in votes.items()
    }


def kept_codes(ann: pd.DataFrame, min_mentions: int = config.MIN_CODE_MENTIONS) -> list:
    """Closed label set (D1): codes with enough annotator mentions, sorted for stable indexing."""
    counts = ann["icd10cm_code"].value_counts()
    return sorted(counts[counts >= min_mentions].index.tolist())


def patient_split(visits: pd.DataFrame, test_fraction=config.TEST_FRACTION, seed=config.SEED):
    """Split visit_ids by patient_id (E2). Returns (train_visit_ids, test_visit_ids) as sets."""
    patients = visits["patient_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_test = int(round(len(patients) * test_fraction))
    test_patients = set(patients[:n_test])
    train_ids, test_ids = set(), set()
    for vid, pid in zip(visits["visit_id"], visits["patient_id"]):
        (test_ids if pid in test_patients else train_ids).add(vid)
    return train_ids, test_ids


def label_matrix(visit_ids, gold: dict, codes: list) -> np.ndarray:
    """Multi-hot gold matrix [n_visits, n_codes], restricted to kept codes."""
    idx = {c: i for i, c in enumerate(codes)}
    Y = np.zeros((len(visit_ids), len(codes)), dtype=np.int8)
    for r, v in enumerate(visit_ids):
        for c in gold.get(v, ()):
            if c in idx:
                Y[r, idx[c]] = 1
    return Y


if __name__ == "__main__":
    visits = load_visits()
    ann = load_annotations()
    gold = build_gold(ann)
    codes = kept_codes(ann)
    train_ids, test_ids = patient_split(visits)
    # sanity: patient disjointness
    p_train = set(visits[visits.visit_id.isin(train_ids)].patient_id)
    p_test = set(visits[visits.visit_id.isin(test_ids)].patient_id)
    print(f"visits={len(visits)}  kept_codes={len(codes)}")
    print(f"train visits={len(train_ids)}  test visits={len(test_ids)}")
    print(f"patient overlap train/test = {len(p_train & p_test)}  (must be 0)")
    gold_labels = sum(len(gold[v]) for v in visits.visit_id if v in gold)
    print(f"total gold (visit,code) pairs = {gold_labels}")
