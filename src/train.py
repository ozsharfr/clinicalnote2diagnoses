"""Build-time entrypoint: fit the model and save artifacts (model.pkl + emb cache).

Run once at Docker build (internet allowed) so inference is fully offline.
    python -m src.train
"""
from . import data
from .model import DiagnosisModel


def main():
    visits = data.load_visits()
    ann = data.load_annotations()
    model = DiagnosisModel().fit(visits, ann)
    path = model.save()
    print(f"saved model -> {path}  (tau={model.tau:.2f}, codes={len(model.codes)})")


if __name__ == "__main__":
    main()
