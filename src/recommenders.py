import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity,
)

class RecommendationError(
    ValueError
):
    """User-facing recommendation error with an i18n key."""

    def __init__(
        self,
        message_key: str,
    ):
        self.message_key = (
            message_key
        )

        super().__init__(
            message_key
        )

HOTEL_DISPLAY_COLUMNS = [
    "hotel_id",
    "hotel_name",
    "hotel_rank",
    "total_score",
    "hotel_address",
]


# Rule search
STAR_PATTERN = re.compile(
    r"\b([1-5](?:[\.,]5)?)\s*"
    r"(?:sao|stars?)\b",
    flags=re.IGNORECASE,
)


PROPERTY_TYPE_ALIASES = [
    (
        "apartment",
        [
            "can ho",
            "apartment",
            "condo",
            "condotel",
        ],
    ),
    (
        "villa",
        [
            "biet thu",
            "villa",
        ],
    ),
    (
        "homestay",
        [
            "homestay",
        ],
    ),
    (
        "house",
        [
            "nha rieng",
            "house",
        ],
    ),
    (
        "resort",
        [
            "resort",
            "khu nghi duong",
        ],
    ),
    (
        "hotel",
        [
            "khach san",
            "hotel",
        ],
    ),
]


LOCATION_TERMS = [
    "cam hai dong",
    "bai dai",
    "loc tho",
    "vinh hai",
    "vinh phuoc",
    "xuong huan",
    "phuoc long",
    "vinh truong",
    "cam nghia",
    "cam ranh",
    "nha trang",
]


# Helper bỏ dấu để parse query
def normalize_for_matching(
    value,
) -> str:
    """Normalize text for rule-based matching."""

    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )

    text = text.replace(
        "đ",
        "d",
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# Helper rank token
def make_rank_token(
    value: float,
) -> str:
    rank_value = (
        f"{float(value):g}"
        .replace(".", "_")
    )

    return f"rank{rank_value}"


# Extract hạng sao từ query
def extract_star_rating(
    query: str,
):
    normalized_query = (
        query
        .lower()
        .replace("-", " ")
    )

    match = STAR_PATTERN.search(
        normalized_query
    )

    if match is None:
        return None

    value = (
        match.group(1)
        .replace(",", ".")
    )

    return float(value)


# Extract loại hình khách sạn
def extract_property_type(
    query: str,
):
    query_text = (
        normalize_for_matching(
            query
        )
    )

    for (
        property_type,
        aliases,
    ) in PROPERTY_TYPE_ALIASES:

        if any(
            alias in query_text
            for alias in aliases
        ):
            return property_type

    return None


# Phân loại từng hotel
def infer_property_type(
    hotel_name,
) -> str:
    name = normalize_for_matching(
        hotel_name
    )

    if (
        "can ho" in name
        or "apartment" in name
        or "condo" in name
        or "condotel" in name
    ):
        return "apartment"

    if (
        "biet thu" in name
        or "villa" in name
    ):
        return "villa"

    if "homestay" in name:
        return "homestay"

    if (
        "nha rieng" in name
        or "house" in name
    ):
        return "house"

    if (
        "resort" in name
        or "khu nghi duong" in name
    ):
        return "resort"

    return "hotel"


# Extract location
def extract_location(
    query: str,
):
    query_text = (
        normalize_for_matching(
            query
        )
    )

    for location in LOCATION_TERMS:
        if location in query_text:
            return location

    return None


# Query expansion
def expand_search_query(
    query: str,
) -> str:
    """Add simple synonyms for TF-IDF search."""

    query_text = (
        query
        .strip()
        .lower()
        .replace("-", " ")
    )

    star_rating = (
        extract_star_rating(
            query_text
        )
    )

    if star_rating is not None:
        query_text = STAR_PATTERN.sub(
            make_rank_token(
                star_rating
            ),
            query_text,
        )

    match_text = (
        normalize_for_matching(
            query
        )
    )

    extra_terms = []

    if (
        "resort" in match_text
        or "khu nghi duong"
        in match_text
    ):
        extra_terms.append(
            "resort khu nghỉ dưỡng"
        )

    if (
        "can ho" in match_text
        or "apartment"
        in match_text
    ):
        extra_terms.append(
            "căn hộ apartment"
        )

    if (
        "biet thu" in match_text
        or "villa" in match_text
    ):
        extra_terms.append(
            "biệt thự villa"
        )

    if (
        "gan bien" in match_text
        or "beach" in match_text
    ):
        extra_terms.append(
            "gần biển bãi biển "
            "ven biển beach beachfront"
        )

    if (
        "gan san bay" in match_text
        or "airport" in match_text
    ):
        extra_terms.append(
            "gần sân bay sân bay airport"
        )

    if "cao cap" in match_text:
        extra_terms.append(
            "cao cấp sang trọng luxury"
        )

    expanded_query = (
        query_text
        + " "
        + " ".join(extra_terms)
    )

    return expanded_query.strip()


# Candidate filtering
def get_candidate_rows(
    query: str,
    hotel_info: pd.DataFrame,
):
    property_type = (
        extract_property_type(
            query
        )
    )

    star_rating = (
        extract_star_rating(
            query
        )
    )

    location = (
        extract_location(
            query
        )
    )

    candidate_mask = pd.Series(
        True,
        index=hotel_info.index,
    )

    # --------------------------------------------------------
    # Property type
    # --------------------------------------------------------

    if property_type is not None:

        hotel_types = (
            hotel_info["hotel_name"]
            .map(
                infer_property_type
            )
        )

        candidate_mask &= (
            hotel_types.eq(
                property_type
            )
        )

    # --------------------------------------------------------
    # Star rating
    # --------------------------------------------------------

    if star_rating is not None:

        candidate_mask &= (
            hotel_info["hotel_rank"]
            .eq(star_rating)
            .fillna(False)
        )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    if location is not None:

        location_text = (
            hotel_info[
                "hotel_name"
            ].fillna("")
            + " "
            + hotel_info[
                "hotel_address"
            ].fillna("")
        ).map(
            normalize_for_matching
        )

        candidate_mask &= (
            location_text
            .str.contains(
                location,
                regex=False,
                na=False,
            )
        )

    candidate_rows = (
        hotel_info.loc[
            candidate_mask,
            "content_row",
        ]
        .astype(int)
        .to_numpy()
    )

    return candidate_rows


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
        raise RecommendationError(
            "error.hotel_not_found"
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
        raise RecommendationError(
            "error.query_required"
        )

    expanded_query = (
        expand_search_query(
            query
        )
    )

    query_vector = (
        tfidf_vectorizer.transform(
            [expanded_query]
        )
    )

    if query_vector.nnz == 0:
        raise RecommendationError(
            "error.no_matching_keywords"
        )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix,
    ).ravel()

    candidate_rows = (
        get_candidate_rows(
            query=query,
            hotel_info=hotel_info,
        )
    )

    if len(candidate_rows) == 0:
        raise RecommendationError(
            "error.no_candidates"
        )

    ranked_rows = candidate_rows[
        np.argsort(
            scores[
                candidate_rows
            ]
        )[::-1]
    ]

    ranked_rows = [
        int(row)
        for row in ranked_rows
        if scores[row] > 0
    ][:top_n]

    if not ranked_rows:
        raise RecommendationError(
            "error.no_content_match"
        )

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
        raise RecommendationError(
            "error.no_gensim_recommendation"
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
        raise RecommendationError(
            "error.customer_group_not_found"
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