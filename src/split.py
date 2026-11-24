from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .paths import PROC_DIR


def make_query_splits(
    queries: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    q_ids = queries["query_id"].drop_duplicates().astype("int64")

    train_val_ids, test_ids = train_test_split(
        q_ids,
        test_size=test_size,
        random_state=random_state,
    )

    rel_val = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=rel_val,
        random_state=random_state,
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
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    queries = pd.read_parquet(proc_dir / "queries.parquet")
    splits = make_query_splits(
        queries,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    splits.to_parquet(proc_dir / "splits.parquet", index=False)
    return splits