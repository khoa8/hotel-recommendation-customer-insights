import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity,
)


HOTEL_DISPLAY_COLUMNS = [
    "hotel_id",
    "hotel_name",
    "hotel_rank",
    "total_score",
    "hotel_address",
]


def recommend_by_cosine(
    hotel_id: str,
    hotel_info: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend similar hotels with TF-IDF cosine similarity."""

    selected = hotel_info.loc[
        hotel_info["hotel_id"].eq(hotel_id)
    ]

    if selected.empty:
        raise ValueError(
            "Không tìm thấy khách sạn."
        )

    source_row = int(
        selected["content_row"].iloc[0]
    )

    scores = cosine_similarity(
        tfidf_matrix[source_row],
        tfidf_matrix,
    ).ravel()

    ranked_rows = np.argsort(scores)[::-1]

    ranked_rows = [
        int(row)
        for row in ranked_rows
        if int(row) != source_row
    ][:top_n]

    catalog_by_row = hotel_info.set_index(
        "content_row",
        drop=False,
    )

    result = (
        catalog_by_row
        .loc[ranked_rows, HOTEL_DISPLAY_COLUMNS]
        .copy()
        .reset_index(drop=True)
    )

    result.insert(
        2,
        "similarity_score",
        scores[ranked_rows],
    )

    result.insert(
        0,
        "rank",
        range(1, len(result) + 1),
    )

    return result


def recommend_by_query(
    query: str,
    hotel_info: pd.DataFrame,
    tfidf_matrix,
    tfidf_vectorizer,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend hotels from a free-text query."""

    query = query.strip()

    if not query:
        raise ValueError(
            "Vui lòng nhập nhu cầu tìm khách sạn."
        )

    query_vector = (
        tfidf_vectorizer.transform(
            [query]
        )
    )

    if query_vector.nnz == 0:
        raise ValueError(
            "Không tìm thấy từ khóa phù hợp "
            "trong dữ liệu khách sạn. "
            "Hãy thử mô tả khác."
        )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix,
    ).ravel()

    ranked_rows = (
        np.argsort(scores)[::-1]
    )

    ranked_rows = [
        int(row)
        for row in ranked_rows
        if scores[row] > 0
    ][:top_n]

    catalog_by_row = (
        hotel_info.set_index(
            "content_row",
            drop=False,
        )
    )

    result = (
        catalog_by_row
        .loc[
            ranked_rows,
            HOTEL_DISPLAY_COLUMNS,
        ]
        .copy()
        .reset_index(drop=True)
    )

    result.insert(
        2,
        "similarity_score",
        scores[ranked_rows],
    )

    result.insert(
        0,
        "rank",
        range(
            1,
            len(result) + 1,
        ),
    )

    return result


def recommend_by_gensim(
    hotel_id: str,
    hotel_info: pd.DataFrame,
    gensim_top20: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Read precomputed Gensim recommendations."""

    recommendations = (
        gensim_top20.loc[
            gensim_top20[
                "source_hotel_id"
            ].eq(hotel_id)
        ]
        .sort_values("rank")
        .head(top_n)
        .copy()
    )

    if recommendations.empty:
        raise ValueError(
            "Không có Gensim recommendation "
            "cho khách sạn này."
        )

    recommendations = recommendations.merge(
        hotel_info[HOTEL_DISPLAY_COLUMNS],
        left_on="recommended_hotel_id",
        right_on="hotel_id",
        how="left",
    )

    return recommendations[
        [
            "rank",
            "hotel_id",
            "similarity_score",
            "hotel_name",
            "hotel_rank",
            "total_score",
            "hotel_address",
        ]
    ].reset_index(drop=True)


def recommend_with_ridge(
    customer_group: str,
    hotel_info: pd.DataFrame,
    cf_data: pd.DataFrame,
    ridge_model,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend unseen hotels for a customer group."""

    valid_groups = set(
        cf_data["customer_group"]
    )

    if customer_group not in valid_groups:
        raise ValueError(
            "Không tìm thấy customer group."
        )

    seen_hotels = set(
        cf_data.loc[
            cf_data["customer_group"].eq(
                customer_group
            ),
            "hotel_id",
        ]
    )

    all_candidate_hotels = (
        cf_data["hotel_id"]
        .drop_duplicates()
        .tolist()
    )

    candidate_hotels = [
        hotel_id
        for hotel_id in all_candidate_hotels
        if hotel_id not in seen_hotels
    ]

    candidates = pd.DataFrame(
        {
            "customer_group": customer_group,
            "hotel_id": candidate_hotels,
        }
    )

    feature_columns = [
        "customer_group",
        "hotel_id",
    ]

    candidates["prediction"] = np.clip(
        ridge_model.predict(
            candidates[feature_columns]
        ),
        0,
        10,
    )

    recommendations = (
        candidates
        .sort_values(
            "prediction",
            ascending=False,
        )
        .head(top_n)
        .merge(
            hotel_info[
                HOTEL_DISPLAY_COLUMNS
            ],
            on="hotel_id",
            how="left",
        )
        .reset_index(drop=True)
    )

    recommendations.insert(
        0,
        "rank",
        range(1, len(recommendations) + 1),
    )

    return recommendations