"""Phase 3 — evaluation (the graded centerpiece, D4).

Reported on the held-out TEST set, always against the HUMAN CEILING computed in
the SAME aggregation (apples-to-apples). Two granularities:
  - exact    : full ICD code
  - category : 3-char prefix (isolates the specificity noise; see eda_findings.md)
Metrics: micro + macro Precision/Recall/F1. NOT accuracy (TN-dominated),
NOT ROC-AUC (TN-inflated under ~60 negatives/visit).
"""
from collections import defaultdict
import numpy as np

from . import data, config
from .model import DiagnosisModel


def _cat(c):
    return c.split(".")[0]


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_sets(pred_sets, gold_sets, level="exact"):
    """pred_sets/gold_sets: list of code-sets (aligned). Returns dict of micro/macro P/R/F1."""
    proj = (lambda s: {_cat(c) for c in s}) if level == "category" else (lambda s: set(s))
    # micro: pool
    tp = fp = fn = 0
    per_code = defaultdict(lambda: [0, 0, 0])  # code -> [tp, fp, fn]
    for pr, go in zip(pred_sets, gold_sets):
        pr, go = proj(pr), proj(go)
        tp += len(pr & go); fp += len(pr - go); fn += len(go - pr)
        for c in pr & go: per_code[c][0] += 1
        for c in pr - go: per_code[c][1] += 1
        for c in go - pr: per_code[c][2] += 1
    micro_p, micro_r, micro_f = _prf(tp, fp, fn)
    # macro: average P/R/F1 over codes that appear in gold at least once
    gold_codes = {c for g in gold_sets for c in proj(g)}
    ps, rs, fs = [], [], []
    for c in gold_codes:
        p, r, f = _prf(*per_code[c])
        ps.append(p); rs.append(r); fs.append(f)
    return {
        "micro_P": micro_p, "micro_R": micro_r, "micro_F1": micro_f,
        "macro_P": np.mean(ps), "macro_R": np.mean(rs), "macro_F1": np.mean(fs),
    }


def human_ceiling(ann, codes, level="exact"):
    """Each annotator vs the codes agreed by the OTHER two (>=2), restricted to kept codes.
    Pooled the same way as the model metrics."""
    code_set = set(codes)
    by = defaultdict(lambda: defaultdict(set))  # visit -> annotator -> codes
    for v, c, a in zip(ann["visit_id"], ann["icd10cm_code"], ann["annotator_id"]):
        if c in code_set:
            by[v][a].add(c)
    anns = ["DR_A", "DR_B", "DR_C"]
    pred_sets, gold_sets = [], []
    for v, am in by.items():
        if len(am) < 3:
            continue
        for a in anns:
            others = [o for o in anns if o != a]
            cnt = defaultdict(int)
            for o in others:
                for c in am.get(o, ()):
                    cnt[c] += 1
            gold = {c for c, k in cnt.items() if k >= 2}
            pred_sets.append(am.get(a, set()))
            gold_sets.append(gold)
    return score_sets(pred_sets, gold_sets, level)


def _fmt(d):
    return (f"micro P/R/F1 = {d['micro_P']:.3f}/{d['micro_R']:.3f}/{d['micro_F1']:.3f}"
            f"   macro P/R/F1 = {d['macro_P']:.3f}/{d['macro_R']:.3f}/{d['macro_F1']:.3f}")


def main():
    visits = data.load_visits()
    ann = data.load_annotations()
    gold = data.build_gold(ann)
    codes = data.kept_codes(ann)
    _train_ids, test_ids = data.patient_split(visits)
    test_ids = sorted(test_ids)

    model = DiagnosisModel.load()
    preds = model.predict(visits, test_ids)
    pred_sets = [{it["code"] for it in preds[v]} for v in test_ids]
    gold_sets = [gold.get(v, set()) & set(codes) for v in test_ids]

    print(f"TEST set: {len(test_ids)} visits, {len(codes)} codes, tau={model.tau:.2f}\n")
    for level in ("exact", "category"):
        m = score_sets(pred_sets, gold_sets, level)
        h = human_ceiling(ann, codes, level)
        print(f"[{level:8}]")
        print(f"  MODEL   {_fmt(m)}")
        print(f"  CEILING {_fmt(h)}")
        print(f"  model micro-F1 = {100*m['micro_F1']/h['micro_F1']:.0f}% of human ceiling\n")


if __name__ == "__main__":
    main()
