import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
)


CRITERIA = {
    "location": "Vị trí",
    "cleanliness": "Sạch sẽ",
    "service": "Dịch vụ",
    "facilities": "Tiện nghi",
    "value_for_money": "Đáng giá tiền",
    "comfort_and_room_quality": (
        "Thoải mái & chất lượng phòng"
    ),
}


VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "có", "cho", "với",
    "trong", "tại", "một", "các", "được",
    "những", "này", "đến", "từ", "khi",
    "đã", "rất", "thì", "mình", "tôi",
    "chúng", "bạn", "ở", "không", "cũng",
    "nên", "sẽ", "về", "nhưng", "vì",
    "nhiều", "hơn", "lại", "đây", "đó",
    "nếu", "để", "như", "ra", "vào",
    "trên", "dưới", "cả",
}


GENERIC_STOPWORDS = {
    "khách", "sạn", "hotel", "resort",
    "tuyệt", "vời",
}


def get_hotel_and_reviews(
    hotel_id: str,
    hotel_info: pd.DataFrame,
    hotel_comments: pd.DataFrame,
):
    selected_hotel = hotel_info.loc[
        hotel_info["hotel_id"].eq(hotel_id)
    ]

    if selected_hotel.empty:
        raise ValueError(
            "Không tìm thấy thông tin khách sạn."
        )

    selected_reviews = hotel_comments.loc[
        hotel_comments["hotel_id"].eq(
            hotel_id
        )
    ].copy()

    if selected_reviews.empty:
        raise ValueError(
            "Khách sạn chưa có review."
        )

    selected_reviews["review_date"] = (
        pd.to_datetime(
            selected_reviews["review_date"],
            errors="coerce",
        )
    )

    return (
        selected_hotel.iloc[0],
        selected_reviews,
    )


def _format_date(value) -> str:
    if pd.isna(value):
        return "Không có dữ liệu"

    return value.date().isoformat()


def build_overview_table(
    hotel,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    first_date = reviews[
        "review_date"
    ].min()

    latest_date = reviews[
        "review_date"
    ].max()

    return pd.DataFrame(
        {
            "Thông tin": [
                "Tên khách sạn",
                "Hotel ID",
                "Hạng sao",
                "Địa chỉ",
                "Total Score",
                "Điểm review trung bình",
                "Số review thực tế",
                "comments_count",
                "Review đầu tiên",
                "Review gần nhất",
            ],
            "Giá trị": [
                hotel["hotel_name"],
                hotel["hotel_id"],
                hotel["hotel_rank"],
                hotel["hotel_address"],
                hotel["total_score"],
                round(
                    reviews["score"].mean(),
                    2,
                ),
                len(reviews),
                hotel["comments_count"],
                _format_date(first_date),
                _format_date(latest_date),
            ],
        }
    )


def build_score_level_summary(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        reviews
        .dropna(subset=["score_level"])
        .groupby("score_level")
        .size()
        .sort_values(ascending=False)
        .rename("review_count")
        .reset_index()
    )

    total = result[
        "review_count"
    ].sum()

    result["percentage"] = (
        result["review_count"]
        / total
        * 100
    ).round(2)

    return result


def build_benchmark_table(
    hotel,
    hotel_info: pd.DataFrame,
) -> pd.DataFrame:
    hotel_id = hotel["hotel_id"]

    other_hotels = hotel_info.loc[
        ~hotel_info["hotel_id"].eq(
            hotel_id
        )
    ]

    same_rank_hotels = other_hotels.loc[
        other_hotels["hotel_rank"].eq(
            hotel["hotel_rank"]
        )
    ]

    rows = []

    for column, label in CRITERIA.items():
        hotel_score = hotel[column]
        system_average = (
            other_hotels[column].mean()
        )
        same_rank_average = (
            same_rank_hotels[column].mean()
        )

        if pd.isna(hotel_score):
            difference = np.nan
            status = "Không có dữ liệu"
        else:
            difference = (
                hotel_score
                - system_average
            )

            if difference >= 0.3:
                status = "Điểm mạnh"
            elif difference <= -0.3:
                status = "Cần cải thiện"
            else:
                status = "Gần trung bình"

        rows.append(
            {
                "Tiêu chí": label,
                "Điểm khách sạn": hotel_score,
                "TB toàn hệ thống": (
                    system_average
                ),
                "TB cùng hạng sao": (
                    same_rank_average
                ),
                "Chênh lệch với hệ thống": (
                    difference
                ),
                "Nhận xét": status,
            }
        )

    return pd.DataFrame(rows)


def build_nationality_summary(
    reviews: pd.DataFrame,
    min_reviews: int = 20,
    top_n: int = 10,
) -> pd.DataFrame:
    result = (
        reviews
        .dropna(subset=["nationality"])
        .groupby("nationality")
        .agg(
            review_count=("score", "size"),
            average_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(
            "review_count",
            ascending=False,
        )
    )

    filtered_result = result.loc[
        result["review_count"] >= min_reviews
    ]

    # Nếu khách sạn có quá ít review và không nhóm nào
    # đạt ngưỡng, vẫn hiển thị dữ liệu hiện có.
    if filtered_result.empty and not result.empty:
        filtered_result = result

    filtered_result = (
        filtered_result
        .head(top_n)
        .reset_index(drop=True)
    )

    filtered_result["average_score"] = (
        filtered_result["average_score"].round(2)
    )

    return filtered_result


def build_group_summary(
    reviews: pd.DataFrame,
    min_reviews: int = 20,
) -> pd.DataFrame:
    result = (
        reviews
        .dropna(subset=["group_name"])
        .groupby("group_name")
        .agg(
            review_count=("score", "size"),
            average_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(
            "review_count",
            ascending=False,
        )
    )

    filtered_result = result.loc[
        result["review_count"] >= min_reviews
    ]

    # Fallback cho khách sạn có ít review.
    if filtered_result.empty and not result.empty:
        filtered_result = result

    filtered_result = (
        filtered_result
        .reset_index(drop=True)
    )

    filtered_result["average_score"] = (
        filtered_result["average_score"].round(2)
    )

    return filtered_result


def build_yearly_summary(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    valid_reviews = reviews.dropna(
        subset=["review_date"]
    ).copy()

    valid_reviews["review_year"] = (
        valid_reviews["review_date"].dt.year
    )

    result = (
        valid_reviews
        .groupby("review_year")
        .agg(
            review_count=("score", "size"),
            average_score=("score", "mean"),
        )
        .reset_index()
        .sort_values("review_year")
    )

    result["average_score"] = (
        result["average_score"].round(2)
    )

    return result


def _clean_keyword_text(
    text_series: pd.Series,
) -> pd.Series:
    return (
        text_series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"[^\w\s]",
            " ",
            regex=True,
        )
        .str.replace(
            r"_+",
            " ",
            regex=True,
        )
        .str.replace(
            r"\d+",
            " ",
            regex=True,
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
        .drop_duplicates()
    )


def get_top_keywords(
    text_series: pd.Series,
    hotel_name: str,
    top_n: int = 15,
) -> pd.DataFrame:
    clean_texts = _clean_keyword_text(
        text_series
    )

    clean_texts = clean_texts.loc[
        clean_texts.str.len() > 0
    ]

    if clean_texts.empty:
        return pd.DataFrame(
            columns=[
                "keyword",
                "frequency",
            ]
        )

    hotel_name_words = set(
        re.findall(
            r"\w+",
            hotel_name.lower(),
            flags=re.UNICODE,
        )
    )

    stopwords = list(
        set(ENGLISH_STOP_WORDS)
        | VIETNAMESE_STOPWORDS
        | GENERIC_STOPWORDS
        | hotel_name_words
    )

    for min_df in [2, 1]:
        try:
            vectorizer = CountVectorizer(
                stop_words=stopwords,
                ngram_range=(1, 2),
                min_df=min_df,
            )

            matrix = vectorizer.fit_transform(
                clean_texts
            )

            frequencies = np.asarray(
                matrix.sum(axis=0)
            ).ravel()

            result = pd.DataFrame(
                {
                    "keyword": (
                        vectorizer
                        .get_feature_names_out()
                    ),
                    "frequency": frequencies,
                }
            )

            return (
                result
                .nlargest(
                    top_n,
                    "frequency",
                )
                .reset_index(drop=True)
            )

        except ValueError:
            continue

    return pd.DataFrame(
        columns=[
            "keyword",
            "frequency",
        ]
    )


def build_keyword_tables(
    reviews: pd.DataFrame,
    hotel_name: str,
    top_n: int = 15,
):
    keyword_text = (
        reviews["title"].fillna("")
        + " "
        + reviews["body"].fillna("")
    )

    high_mask = reviews["score"] >= 9
    lower_mask = reviews["score"] < 9

    positive_keywords = get_top_keywords(
        keyword_text.loc[high_mask],
        hotel_name=hotel_name,
        top_n=top_n,
    )

    improvement_keywords = get_top_keywords(
        keyword_text.loc[lower_mask],
        hotel_name=hotel_name,
        top_n=top_n,
    )

    counts = {
        "high_score_reviews": int(
            high_mask.sum()
        ),
        "lower_score_reviews": int(
            lower_mask.sum()
        ),
    }

    return (
        positive_keywords,
        improvement_keywords,
        counts,
    )


def get_lowest_reviews(
    reviews: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    columns = [
        "score",
        "nationality",
        "group_name",
        "title",
        "body",
    ]

    return (
        reviews
        .sort_values("score")
        .drop_duplicates(
            subset=["title", "body"]
        )
        .head(top_n)[columns]
        .reset_index(drop=True)
    )