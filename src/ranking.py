from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from .evaluation import evaluate_ranking


NON_FEATURE_COLS = {"query_id", "product_id", "relevance", "score", "rank"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric/boolean ranking feature columns excluding IDs, targets, and rank/score outputs."""
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [
        c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError("No valid numeric feature columns found in ranking DataFrame.")
    return numeric_cols


def _prepare_grouped_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    require_relevance: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None, list[int]]:
    """Sort by query_id deterministically and compute group sizes for LightGBM."""
    if "query_id" not in df.columns:
        raise ValueError("DataFrame must include 'query_id'.")
    if require_relevance and "relevance" not in df.columns:
        raise ValueError("DataFrame must include 'relevance' for supervised ranking.")

    ordered = df.sort_values(["query_id", "product_id"] if "product_id" in df.columns else ["query_id"])
    ordered = ordered.reset_index(drop=True)

    group_sizes = ordered.groupby("query_id", sort=False).size().astype(int).tolist()
    x = ordered[feature_cols].copy()
    y = ordered["relevance"].to_numpy(dtype=np.int64) if require_relevance else None
    return x, y, group_sizes


def train_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
) -> tuple[LGBMRanker, list[str]]:
    """Train a query-grouped LightGBM ranker on train_df (with optional val_df evaluation)."""
    cols = get_feature_columns(train_df) if feature_cols is None else feature_cols

    x_train, y_train_raw, train_groups_raw = _prepare_grouped_data(
        train_df, cols, require_relevance=True
    )
    if y_train_raw is None:
        raise ValueError("Training data requires relevance labels.")

    print(x_train.dtypes[x_train.dtypes == "object"])
    print("Any inf in X:", np.isinf(x_train.select_dtypes(include=[np.number])).any().any())

    y_train = np.ascontiguousarray(y_train_raw, dtype=np.int32)
    train_groups = np.ascontiguousarray(train_groups_raw, dtype=np.int32)

    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    ranker = LGBMRanker(**default_params)

    fit_kwargs: dict = {
        "X": x_train,
        "y": y_train,
        "group": train_groups,
    }

    if val_df is not None and not val_df.empty:
        x_val, y_val_raw, val_groups_raw = _prepare_grouped_data(
            val_df, cols, require_relevance=True
        )
        if y_val_raw is not None:
            fit_kwargs["eval_X"] = x_val
            fit_kwargs["eval_y"] = np.ascontiguousarray(y_val_raw, dtype=np.int32)
            fit_kwargs["eval_group"] = [np.ascontiguousarray(val_groups_raw, dtype=np.int32)]

    ranker.fit(**fit_kwargs)
    return ranker, cols


def score_candidates(
    ranker: LGBMRanker,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Predict ranking scores per candidate row and sort descending within each query_id."""
    x, _, _ = _prepare_grouped_data(df, feature_cols, require_relevance=False)
    scored = df.sort_values(["query_id", "product_id"] if "product_id" in df.columns else ["query_id"]).reset_index(drop=True)
    scored = scored.copy()
    scored["score"] = ranker.predict(x).astype(np.float64)
    scored = scored.sort_values(["query_id", "score"], ascending=[True, False]).reset_index(drop=True)
    scored["rank"] = scored.groupby("query_id", sort=False).cumcount() + 1
    return scored


def save_ranker(ranker: LGBMRanker, feature_cols: list[str], path: str | Path) -> Path:
    """Persist trained ranker and feature column list to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": ranker, "feature_cols": feature_cols}, out_path)
    return out_path


def load_ranker(path: str | Path) -> tuple[LGBMRanker, list[str]]:
    """Load persisted ranker and feature column list from disk."""
    payload = joblib.load(Path(path))
    return payload["model"], payload["feature_cols"]