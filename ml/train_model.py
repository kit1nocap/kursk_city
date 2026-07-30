"""Train and persist the KursMetr apartment price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "kursk_apartments_cian_snapshot.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "ml" / "artifacts" / "model.joblib"

NUMERIC_FEATURES = ["rooms", "area", "floor", "total_floors"]
CATEGORICAL_FEATURES = ["district", "listing_kind", "address"]
TARGET = "price"

ALIASES = {
    "адрес": "address",
    "район": "district",
    "округ": "district",
    "тип_рынка": "listing_kind",
    "комнат": "rooms",
    "комнаты": "rooms",
    "площадь": "area",
    "этаж": "floor",
    "этажей": "total_floors",
    "этажей_в_доме": "total_floors",
    "цена": "price",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Kursk apartment price model on the bundled dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"JSON or CSV file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Model artifact path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as source:
            rows = json.load(source)
        frame = pd.DataFrame(
            rows,
            columns=[
                "listing_id",
                "address",
                "district",
                "listing_kind",
                "rooms",
                "area",
                "floor",
                "total_floors",
                "price",
            ],
        )
    else:
        frame = pd.read_csv(path)
    frame.columns = [
        ALIASES.get(str(column).strip().lower(), str(column).strip().lower())
        for column in frame.columns
    ]

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["address"] = frame["address"].astype("string").str.strip()
    frame["district"] = frame["district"].astype("string").str.strip()
    frame["listing_kind"] = frame["listing_kind"].astype("string").str.strip()
    frame = frame.dropna(subset=["address", "area", "price"])
    frame = frame[(frame["area"] >= 15) & (frame["area"] <= 300)]
    frame = frame[(frame["price"] >= 500_000) & (frame["price"] <= 100_000_000)]

    if len(frame) < 25:
        raise ValueError(
            f"Only {len(frame)} valid rows remain; at least 25 are required."
        )
    return frame


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    regressor = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", regressor)])


def main() -> None:
    args = parse_args()
    frame = load_dataset(args.data.resolve())
    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = frame[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "target": TARGET,
            "dataset": str(args.data.resolve()),
            "metrics": metrics,
        },
        args.output,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
