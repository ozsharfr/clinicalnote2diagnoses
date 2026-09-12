"""Phase 2 — model.

Multi-label one-vs-rest logistic regression on [63 similarity + metadata] features.
StandardScaler first (E5 similarities sit in a narrow anisotropic band; scaling spreads
them so the linear layer can separate). Global tau calibrated on a patient-disjoint
validation slice to maximize micro-F1, plus a top-1 floor (D5).

    train_core ─┬─► FeatureBuilder.fit (prototypes) ─► scale ─► OvR LogReg
                │
        val ────┴─► calibrate tau (max micro-F1, with top-1 floor)
    then refit on (train_core ∪ val) for the shipped artifact; test stays untouched.
"""
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from . import config, data
from .features import FeatureBuilder


def _micro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold.astype(bool)).sum())
    fn = int((~pred.astype(bool) & gold).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _apply_floor(scores: np.ndarray, tau: float) -> np.ndarray:
    """Multi-hot decision: scores>=tau, but always keep each row's argmax (top-1 floor)."""
    pred = (scores >= tau)
    top1 = scores.argmax(axis=1)
    pred[np.arange(scores.shape[0]), top1] = True
    return pred


class _PerCodeClf:
    """One-vs-rest wrapper that degrades gracefully for codes too rare to train."""
    def __init__(self, codes):
        self.codes = codes
        self.models = []  # (kind, obj_or_const)

    def fit(self, X, Y):
        self.models = []
        for i in range(len(self.codes)):
            y = Y[:, i]
            if y.sum() < 2 or y.sum() == len(y):     # not enough signal to train
                self.models.append(("const", float(y.mean())))
            else:
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(X, y)
                self.models.append(("clf", clf))
        return self

    def predict_scores(self, X) -> np.ndarray:
        out = np.zeros((X.shape[0], len(self.codes)), dtype=np.float32)
        for i, (kind, m) in enumerate(self.models):
            out[:, i] = m if kind == "const" else m.predict_proba(X)[:, 1]
        return out


class DiagnosisModel:
    def __init__(self):
        self.codes = None
        self.fb = None
        self.scaler = None
        self.clf = None
        self.tau = config.DEFAULT_TAU

    # ---- training ----
    def fit(self, visits, ann, verbose=True):
        self.codes = data.kept_codes(ann)
        gold = data.build_gold(ann)
        train_ids, _test_ids = data.patient_split(visits)
        # 3-way: carve a patient-disjoint val slice out of train
        tr_visits = visits[visits.visit_id.isin(train_ids)]
        core_ids, val_ids = data.patient_split(tr_visits, test_fraction=0.2, seed=config.SEED + 1)

        # fit on core, calibrate tau on val
        fb = FeatureBuilder(self.codes).fit(visits, core_ids, gold)
        Xc, _ = fb.transform(visits, sorted(core_ids))
        Xv, _ = fb.transform(visits, sorted(val_ids))
        scaler = StandardScaler().fit(Xc)
        clf = _PerCodeClf(self.codes).fit(scaler.transform(Xc), data.label_matrix(sorted(core_ids), gold, self.codes))

        Yv = data.label_matrix(sorted(val_ids), gold, self.codes)
        Sv = clf.predict_scores(scaler.transform(Xv))
        best_tau, best_f1 = config.DEFAULT_TAU, -1.0
        for tau in np.arange(0.05, 0.96, 0.05):
            f1 = _micro_f1(_apply_floor(Sv, tau), Yv)
            if f1 > best_f1:
                best_tau, best_f1 = float(tau), f1
        if verbose:
            print(f"calibrated tau={best_tau:.2f}  val micro-F1={best_f1:.3f}"
                  f"  (core={len(core_ids)} val={len(val_ids)})")

        # refit on full train (core ∪ val) for the shipped artifact
        self.fb = FeatureBuilder(self.codes).fit(visits, train_ids, gold)
        Xtr, _ = self.fb.transform(visits, sorted(train_ids))
        self.scaler = StandardScaler().fit(Xtr)
        self.clf = _PerCodeClf(self.codes).fit(
            self.scaler.transform(Xtr), data.label_matrix(sorted(train_ids), gold, self.codes))
        self.tau = best_tau
        return self

    # ---- inference ----
    def score_matrix(self, visits, visit_ids):
        X, evidence = self.fb.transform(visits, visit_ids)
        return self.clf.predict_scores(self.scaler.transform(X)), evidence

    def predict(self, visits, visit_ids) -> dict:
        """visit_id -> list of {code, score, evidence, evidence_sim} sorted by score (D5 applied)."""
        scores, evidence = self.score_matrix(visits, visit_ids)
        pred = _apply_floor(scores, self.tau)
        out = {}
        for r, vid in enumerate(visit_ids):
            items = []
            for i in np.where(pred[r])[0]:
                unit, sim = evidence[r][self.codes[i]]
                items.append({"code": self.codes[i], "score": round(float(scores[r, i]), 4),
                              "evidence": unit, "evidence_sim": round(sim, 4)})
            items.sort(key=lambda d: d["score"], reverse=True)
            out[vid] = items
        return out

    # ---- persistence ----
    def save(self, path=config.ARTIFACTS / "model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"codes": self.codes, "prototypes": self.fb.prototypes,
                         "meta": self.fb.meta, "scaler": self.scaler,
                         "clf": self.clf, "tau": self.tau}, f)
        return path

    @classmethod
    def load(cls, path=config.ARTIFACTS / "model.pkl"):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls()
        m.codes = d["codes"]; m.scaler = d["scaler"]; m.clf = d["clf"]; m.tau = d["tau"]
        m.fb = FeatureBuilder(m.codes)
        m.fb.prototypes = d["prototypes"]; m.fb.meta = d["meta"]
        return m


if __name__ == "__main__":
    visits = data.load_visits()
    ann = data.load_annotations()
    model = DiagnosisModel().fit(visits, ann)
    p = model.save()
    print(f"saved -> {p}")
    # sample prediction
    _, test_ids = data.patient_split(visits)
    vid = sorted(test_ids)[0]
    gold = data.build_gold(ann)
    print(f"\nvisit {vid}  gold={gold.get(vid)}")
    for item in model.predict(visits, [vid])[vid]:
        print(f"  {item['code']:8} score={item['score']:.3f}  ev_sim={item['evidence_sim']:.3f}  {item['evidence'][:55]}")
