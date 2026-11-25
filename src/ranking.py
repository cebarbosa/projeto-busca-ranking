from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src.config import load_config

MODEL_BUILDERS = {
    "hist_gradient_boosting": lambda params: HistGradientBoostingRegressor(**params),
    "ridge": lambda params: make_pipeline(SimpleImputer(strategy="median"), Ridge(**params)),
}


NON_FEATURE_COLS = {
    "query_id",
    "product_id",
    "label",
    "relevance",
    "split",
    "query",
    "title",
    "product_title",
    "description",
    "product_description",
    "category",
    "product_category",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric/boolean feature columns excluding identifiers, text, and targets."""
    cols: list[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No valid feature columns found in ranking DataFrame.")
    return cols


def _prepare_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Convert selected feature columns into a clean float64 numpy array."""
    x = df[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(x[col]):
            x[col] = x[col].astype(np.float64)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(np.float64)
    return np.ascontiguousarray(x.to_numpy(dtype=np.float64))


def train_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    target_col: str = "relevance",
    model_name: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[Any, list[str], dict[str, Any]]:
    cfg = cfg or load_config()
    ranking_cfg = cfg["ranking"]
    selected_name = model_name or ranking_cfg["active_model"]
    model_cfg = ranking_cfg["models"][selected_name]

    cols = feature_cols or get_feature_columns(train_df)
    model = build_ranker(model_cfg)
    model.fit(train_df[cols], train_df[target_col])

    meta = {
        "model_name": selected_name,
        "model_type": model_cfg["type"],
        "params": model_cfg.get("params", {}),
    }
    return model, cols, meta


def score_candidates(
    ranker: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Attach predicted ranking scores to candidate rows."""
    scored = df.copy()
    if scored.empty:
        scored["score"] = pd.Series(dtype="float64")
        return scored

    x_mat = _prepare_matrix(scored, feature_cols)
    scored["score"] = np.asarray(ranker.predict(x_mat), dtype=np.float64)
    return scored


def save_ranker(
    model: Any,
    feature_cols: list[str],
    output_path: Path | str,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "meta": meta or {},
    }
    joblib.dump(payload, path)
    return path


def load_ranker(artifact_path: Path | str) -> tuple[Any, list[str]]:
    obj = joblib.load(Path(artifact_path))
    if isinstance(obj, dict):
        return obj["model"], obj["feature_cols"]
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[0], obj[1]
    raise ValueError(f"Unrecognized ranker artifact format at {artifact_path}")