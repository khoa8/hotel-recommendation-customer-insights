from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def _require_file(filename: str) -> Path:
    """Return an artifact path or raise a clear error."""
    path = ARTIFACT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy artifact: {path}"
        )

    return path


@st.cache_data
def load_hotel_info() -> pd.DataFrame:
    return pd.read_parquet(
        _require_file(
            "hotel_info_clean.parquet"
        )
    )


@st.cache_data
def load_hotel_comments() -> pd.DataFrame:
    data = pd.read_parquet(
        _require_file(
            "hotel_comments_clean.parquet"
        )
    )

    data["review_date"] = pd.to_datetime(
        data["review_date"],
        errors="coerce",
    )

    return data


@st.cache_data
def load_cf_data() -> pd.DataFrame:
    return pd.read_parquet(
        _require_file("cf_data.parquet")
    )


@st.cache_data
def load_gensim_top20() -> pd.DataFrame:
    return pd.read_parquet(
        _require_file("gensim_top20.parquet")
    )


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(
        _require_file("model_comparison.csv")
    )


@st.cache_data
def load_deployment_summary() -> dict:
    with open(
        _require_file(
            "deployment_summary.json"
        ),
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_resource
def load_tfidf_matrix():
    return sparse.load_npz(
        _require_file("tfidf_matrix.npz")
    )


@st.cache_resource
def load_tfidf_vectorizer():
    return joblib.load(
        _require_file(
            "tfidf_vectorizer.joblib"
        )
    )


@st.cache_resource
def load_ridge_model():
    return joblib.load(
        _require_file(
            "ridge_full_model.joblib"
        )
    )