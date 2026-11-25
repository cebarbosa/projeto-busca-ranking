from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import load_config
from .paths import PROC_DIR


def make_query_splits(
    queries: pd.DataFrame,
    val_size: float | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    config = cfg or load_config()
    split_cfg = config.get("split", {})

    resolved_val_size = (
        float(val_size)
        if val_size is not None
        else float(split_cfg.get("val_ratio", config.get("val_size", 0.15)))
    )
    resolved_test_size = (
        float(test_size)
        if test_size is not None
        else float(split_cfg.get("test_ratio", config.get("test_size", 0.15)))
    )
    resolved_seed = (
        int(random_state)
        if random_state is not None
        else int(split_cfg.get("seed", config.get("seed", 42)))
    )

    q_ids = queries["query_id"].drop_duplicates().astype("int64")

    train_val_ids, test_ids = train_test_split(
        q_ids,
        test_size=resolved_test_size,
        random_state=resolved_seed,
    )

    rel_val = resolved_val_size / (1.0 - resolved_test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=rel_val,
        random_state=resolved_seed,
    )

    splits = pd.concat(
        [
            pd.DataFrame({"query_id": train_ids, "split": "train"}),
            pd.DataFrame({"query_id": val_ids, "split": "val"}),
            pd.DataFrame({"query_id": test_ids, "split": "test"}),
        ],
        ignore_index=True,
    )
    return splits.sort_values("query_id").reset_index(drop=True)


def materialize_splits(
    proc_dir: Path = PROC_DIR,
    val_size: float | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    queries = pd.read_parquet(proc_dir / "queries.parquet")
    splits = make_query_splits(
        queries,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        cfg=cfg,
    )
    splits.to_parquet(proc_dir / "splits.parquet", index=False)
    return splits