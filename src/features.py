"""Phase 1 — features (the core).

Pipeline per visit:
    note ──► segment (newlines) ──► units ──► mE5 embed (cached, normalized)
    for each kept code i:
        feature_i   = max over units of cos(unit, prototype_i)
        evidence_i  = the argmax unit  (D6: evidence falls out of feature construction)
    + metadata one-hot / numeric (D3)

Prototypes (D3-B2, Hebrew<->Hebrew, no translation):
    prototype_i = mean of unit embeddings from TRAIN visits whose gold set has code i.
    Built on TRAIN ONLY (E2 — no leakage).
"""
import hashlib
import pickle
import numpy as np

from . import config


# ---------- segmentation ----------
def segment(note_text: str) -> list:
    """Split a note into units. Newline-based (notes are line-structured, ~5.5 lines);
    period-splitting breaks on Hebrew clinical abbreviations (מ״ג/ד״ל, ל״ד)."""
    units = [ln.strip(" \t-•*") for ln in note_text.replace("\r", "\n").split("\n")]
    units = [u for u in units if len(u) >= 3]
    return units or [note_text.strip()]  # never return empty


# ---------- embedder (mE5, disk-cached — E5 is the time sink) ----------
class Embedder:
    def __init__(self, model_name: str = config.MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._cache_path = config.EMB_CACHE / (hashlib.sha1(model_name.encode()).hexdigest()[:8] + ".pkl")
        self._cache = {}
        if self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def _lazy_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list) -> np.ndarray:
        """Return L2-normalized embeddings [n, dim], using and updating the disk cache."""
        missing = [t for t in texts if t not in self._cache]
        if missing:
            uniq = list(dict.fromkeys(missing))
            vecs = self._lazy_model().encode(
                [config.E5_PREFIX + t for t in uniq],
                normalize_embeddings=True, batch_size=64, show_progress_bar=False,
            )
            for t, v in zip(uniq, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
        return np.vstack([self._cache[t] for t in texts])


# ---------- metadata encoder (fit on TRAIN) ----------
class MetadataEncoder:
    def fit(self, visits_df, train_ids):
        tr = visits_df[visits_df.visit_id.isin(train_ids)]
        self.categories_ = {f: sorted(tr[f].astype(str).unique().tolist()) for f in config.CATEGORICAL_META}
        self.age_median_ = float(tr["age"].median())
        self.columns_ = [f"{f}={v}" for f in config.CATEGORICAL_META for v in self.categories_[f]] + ["age"]
        return self

    def transform(self, visits_df, visit_ids) -> np.ndarray:
        rows = visits_df.set_index("visit_id").loc[list(visit_ids)]
        out = np.zeros((len(visit_ids), len(self.columns_)), dtype=np.float32)
        col = {c: i for i, c in enumerate(self.columns_)}
        for r, (_, row) in enumerate(rows.iterrows()):
            for f in config.CATEGORICAL_META:
                key = f"{f}={row[f]}"
                if key in col:                       # unseen category -> all zeros (fine)
                    out[r, col[key]] = 1.0
            age = row["age"]
            out[r, col["age"]] = self.age_median_ if (age != age) else float(age)  # NaN check
        return out


# ---------- the feature builder ----------
class FeatureBuilder:
    def __init__(self, codes: list):
        self.codes = codes
        self.embedder = Embedder()
        self.meta = MetadataEncoder()
        self.prototypes = None  # [n_codes, dim], L2-normalized

    def fit(self, visits_df, train_ids, gold: dict):
        """Build prototypes from TRAIN units (E2) and fit the metadata encoder."""
        note_by_id = dict(zip(visits_df.visit_id, visits_df.note_text))
        buckets = {c: [] for c in self.codes}         # code -> list of unit embeddings (train only)
        code_set = set(self.codes)
        for vid in train_ids:
            g = gold.get(vid, set()) & code_set
            if not g:
                continue
            units = segment(note_by_id[vid])
            emb = self.embedder.encode(units)          # [u, dim]
            for c in g:
                buckets[c].append(emb)
        dim = next(iter(buckets.values()))[0].shape[1] if any(buckets.values()) else 384
        protos = np.zeros((len(self.codes), dim), dtype=np.float32)
        for i, c in enumerate(self.codes):
            if buckets[c]:
                m = np.vstack(buckets[c]).mean(axis=0)
                n = np.linalg.norm(m)
                protos[i] = m / n if n > 0 else m
        self.prototypes = protos
        self.meta.fit(visits_df, train_ids)
        return self

    def transform(self, visits_df, visit_ids):
        """Return (X [n, n_codes+meta], evidence[list of dict code->(unit_text, sim)])."""
        note_by_id = dict(zip(visits_df.visit_id, visits_df.note_text))
        sim_feats = np.zeros((len(visit_ids), len(self.codes)), dtype=np.float32)
        evidence = []
        for r, vid in enumerate(visit_ids):
            units = segment(note_by_id[vid])
            emb = self.embedder.encode(units)          # [u, dim], normalized
            sims = emb @ self.prototypes.T             # [u, n_codes] cosine (both normalized)
            best_unit = sims.argmax(axis=0)            # per code: which unit
            sim_feats[r] = sims.max(axis=0)
            evidence.append({
                self.codes[i]: (units[best_unit[i]], float(sim_feats[r, i]))
                for i in range(len(self.codes))
            })
        meta_feats = self.meta.transform(visits_df, visit_ids)
        X = np.hstack([sim_feats, meta_feats])
        return X, evidence

    @property
    def feature_names(self):
        return [f"sim::{c}" for c in self.codes] + self.meta.columns_


if __name__ == "__main__":
    from . import data
    visits = data.load_visits()
    ann = data.load_annotations()
    gold = data.build_gold(ann)
    codes = data.kept_codes(ann)
    train_ids, test_ids = data.patient_split(visits)

    fb = FeatureBuilder(codes).fit(visits, train_ids, gold)
    Xtr, ev = fb.transform(visits, sorted(train_ids)[:5])
    print(f"prototypes: {fb.prototypes.shape}  feature dim: {Xtr.shape[1]}"
          f" ({len(codes)} sim + {len(fb.meta.columns_)} meta)")
    # peek: for the first sample, top-3 codes by similarity + their evidence
    v0 = sorted(train_ids)[0]
    top = sorted(ev[0].items(), key=lambda kv: kv[1][1], reverse=True)[:3]
    print(f"\nvisit {v0} gold={gold.get(v0)}")
    for c, (unit, sim) in top:
        print(f"  {c:8} sim={sim:.3f}  evidence: {unit[:70]}")
