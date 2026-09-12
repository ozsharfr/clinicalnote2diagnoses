"""Inference API (D7). Single POST /predict endpoint.

Loads the model once at startup from artifacts/model.pkl. Fully offline at runtime.
    uvicorn api.main:app --port 8000
"""
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src import config
from src.model import DiagnosisModel

app = FastAPI(title="ClinicalNote2Diagnoses", version="0.1")
_model = DiagnosisModel.load()


class Visit(BaseModel):
    visit_id: str = "V0000"
    note_text: str
    age: Optional[float] = None
    sex: Optional[str] = ""
    physician_specialty: Optional[str] = ""
    encounter_type: Optional[str] = ""
    visit_type: Optional[str] = ""
    care_setting: Optional[str] = ""


@app.get("/health")
def health():
    return {"status": "ok", "codes": len(_model.codes), "tau": _model.tau}


@app.post("/predict")
def predict(visit: Visit):
    row = visit.model_dump()
    df = pd.DataFrame([{**row, "age": row.get("age")}])
    for f in config.CATEGORICAL_META:
        df[f] = df[f].fillna("").astype(str)
    preds = _model.predict(df, [visit.visit_id])[visit.visit_id]
    return {"visit_id": visit.visit_id, "predictions": preds}
