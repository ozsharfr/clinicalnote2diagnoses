"""Central config: paths, model, and the knobs behind decisions D1-D7.

See notes/eda_findings.md for the reasoning behind each constant.
"""
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent
VISITS_CSV = ROOT / "visits.csv"
ANNOTATIONS_CSV = ROOT / "physician_annotations.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
EMB_CACHE = ARTIFACTS / "emb_cache"          # cached unit embeddings (E5 is the time sink)
EMB_CACHE.mkdir(exist_ok=True)

# ---- model (E1: small by default; base is a drop-in swap) ----
MODEL_NAME = "intfloat/multilingual-e5-small"   # ~470MB, 384-dim
E5_PREFIX = "query: "                             # e5 needs a prefix; same on both sides = symmetric sim

# ---- D1: closed label space, drop rare ----
MIN_CODE_MENTIONS = 5        # keep codes with >= this many annotator mentions (~63 codes, 99.4% coverage)

# ---- D2: gold = majority ----
MIN_ANNOTATORS_FOR_GOLD = 2  # a (visit, code) is gold if >=2 of 3 annotators gave it

# ---- split (E2: by patient, never leak a patient across train/test) ----
TEST_FRACTION = 0.2
SEED = 42

# ---- D3: metadata features ----
CATEGORICAL_META = ["physician_specialty", "sex", "encounter_type", "visit_type", "care_setting"]
NUMERIC_META = ["age"]

# ---- D5: thresholding ----
# tau is calibrated on validation (see model.py); top-1 floor always emits the best code.
DEFAULT_TAU = 0.5
