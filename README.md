# ClinicalNote2Diagnoses

A small POC that reads a Hebrew clinical visit note and suggests ICD-10-CM diagnosis
codes, each with a confidence score and a supporting span from the note.

## Run it (one command)

```bash
docker compose up --build
```

Then query the API (runs fully offline after build):

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @sample_visit.json
```

Response format:

```json
{
  "visit_id": "T1",
  "predictions": [
    {"code": "E11.9", "score": 0.999, "evidence": "מעקב סוכרת סוג 2, HbA1c 8.1%", "evidence_sim": 0.94}
  ]
}
```

`GET /health` returns model status.

### Run locally without Docker

```bash
pip install -r requirements.txt
python -m src.train          # downloads mE5, fits model -> artifacts/model.pkl
uvicorn api.main:app --port 8000
python -m src.evaluate       # prints the results table below
```

## Results (held-out test set, 120 visits)

Reported against the **human ceiling** — each physician vs the codes agreed by the other
two — because there is no single gold label and even the doctors only agree ~0.86 F1.
The ceiling is computed in the same aggregation as the model (apples-to-apples).

| Granularity | | micro P/R/F1 | macro P/R/F1 | % of human ceiling (micro-F1) |
|---|---|---|---|---|
| **exact** | model | 0.673 / 0.753 / **0.711** | 0.635 / 0.641 / 0.625 | **83%** |
| | human ceiling | 0.800 / 0.923 / 0.857 | 0.551 / 0.613 / 0.579 | |
| **category (3-char)** | model | 0.708 / 0.773 / **0.739** | 0.771 / 0.760 / 0.752 | **82%** |
| | human ceiling | 0.837 / 0.966 / 0.897 | 0.783 / 0.907 / 0.839 | |

Notes:
- The model reaches ~83% of the human ceiling — strong given how much the annotators
  themselves disagree.
- At **exact** level the model's *macro* F1 (0.625) exceeds the human ceiling's macro
  (0.579): `class_weight=balanced` makes it more consistent on less-common codes than
  an individual annotator.
- Metrics are P/R/F1, not accuracy (which is dominated by true negatives — ~60 per visit)
  and not ROC-AUC (same true-negative inflation).

## Approach
1. My first step was to use basic EDA (with the assistance of Claude) in order to:
	a. understand how many doctors involve at each visit
	b. The next question was - in how many cases, diagnosis differ, and in how many it was fairly similar.
	
	It was found that cases that often contradict, are when only the first hierarchy is shown - i.e. 11 instead of 11.9 etc
	Thus - decision was to define "correct"/"gold" as cases where at least 2 doctors agreed 
2. Next step was about removing rare labels - annotations only use 75 distinct codes, and of them, 
	the 63 with at least five mentions cover ~ 99% of all annotations. 
	That was good-enough solution given the time limitation - and decision to focus on the paretto (i.e. highest ROI)

3. Adding embedding-distance based features : 
	a. notes were splitted into lines (next step: further seperat by commas, etc). 
	b. For each line : ME5 embeddings 
	c. for each of the 63 codes a "reference-set" was built - average all vectors of the training-note lines for each of the 63 labels
	d. Now, for all separated sentences : takes minimal distance for each of the 63 labels average vectors. So - we get 63 features with minimal distance for each note
	e. added visit metadata (age, sex, etc)
4. Train the model:
	a. Data was splitted by patients (avoid leakage) into train/val/test - 230/58/72 patients (379/101/120 visits)
	b. Train : Logistic regression, one vs rest - for simplicity and explainability. Trying more complicated algorithm like HistGradientBoosting on the same split did not improve the results.
	c. Optimize probabilty threshold (on val set only) : find the one which provides best micro-f1

5. Evaluation - Micro Precision/Recall/F1 , and also macro at exact and category level. 
	Always compared against doctors  ~0.86 F1, so I read the model relative to that; 
	it lands around 83% of the ceiling. 
	
6. Evidence/explainability - In continue to part 3d above - Each prediction returns the note line that drove its similarity. 


7. Next steps:
	a. Human in the loop - Since the doctors "agreement" is not very high (~0.86 F1), 
		take highest probabilty errors / lowest prob / lowest margins to a doctor/human to re-label
	b. Add explicit numeric features (HbA1c, blood pressure, BMI) for the common codes
	c. Train also where label is higher hierarchy, i.e. 11 instead of 11.9, fo example. 
	   Then combine the results - either in a second model or by rule base



## Assumptions & limitations

- **Closed set**: cannot predict codes outside the 63 seen in the data.
- **Majority gold** discards ~18% single-annotator labels; some are legitimate minority dx.
- **Specificity noise**: annotators split on parent vs leaf codes (E11 vs E11.9); exact-match
  scoring penalizes this, which is why category-level is also reported.
- **mE5 similarity is anisotropic** (values cluster ~0.94); the classifier's StandardScaler
  handles it, but the raw `evidence_sim` is a weak absolute confidence.
- **Segmentation** is newline-based; a single-line note yields one evidence span for all codes.
- **Numeric lab/vital features** (HbA1c, BP, ...) were scoped out under time; a natural next
  improvement for the top metabolic codes.
- **Split**: single patient-level 80/20 split; k-fold CV would give more robust macro numbers.
- **Prototype self-inclusion**: a training visit's own lines are part of the prototypes used
  to build its features, so train features are mildly optimistic (worse for rare codes). Test
  numbers are unaffected (test visits never feed the prototypes); a leave-one-out prototype
  at train time would remove it.

## ICD-10-CM source

FY2026 (October 1, 2025) release, Tabular List XML, U.S. CDC/NCHS.

## Layout

```
src/{config,data,features,model,train,evaluate}.py   # pipeline
api/main.py                                           # FastAPI /predict
Dockerfile  docker-compose.yml  requirements.txt
notes/                                                # design + EDA (not shipped in image)
```
