from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from src.insights import (
    CRITERIA,
    get_highest_reviews,
    get_lowest_reviews,
)

from src.ui_helpers import (
    extract_hotel_highlights,
)

from src.image_helpers import (
    get_hotel_image,
)


def _format_score(
    value,
    decimals: int = 1,
) -> str:
    """Format a numeric score or return N/A."""

    if pd.isna(value):
        return "N/A"

    return f"{float(value):.{decimals}f}"


def _build_maps_url(
    hotel_name: str,
    address: str,
) -> str:
    """Build a Google Maps search URL."""

    query = (
        f"{hotel_name}, {address}"
    )

    encoded_query = quote_plus(
        query
    )

    return (
        "https://www.google.com/maps/"
        "search/?api=1&query="
        f"{encoded_query}"
    )


def _display_review_list(
    reviews: pd.DataFrame,
):
    """Display reviews as user-friendly cards."""

    if reviews.empty:
        st.info(
            "Không có review phù hợp "
            "trong dữ liệu."
        )
        return

    for review in reviews.itertuples(
        index=False
    ):
        with st.container(
            border=True
        ):
            score = _format_score(
                review.score,
                decimals=1,
            )

            title = (
                str(review.title).strip()
                if pd.notna(review.title)
                else ""
            )

            body = (
                str(review.body).strip()
                if pd.notna(review.body)
                else ""
            )

            if title:
                st.markdown(
                    f"**⭐ {score}/10 — "
                    f"{title}**"
                )
            else:
                st.markdown(
                    f"**⭐ {score}/10**"
                )

            if body:
                st.write(body)

            metadata = []

            if pd.notna(
                review.nationality
            ):
                metadata.append(
                    str(
                        review.nationality
                    )
                )

            if pd.notna(
                review.group_name
            ):
                metadata.append(
                    str(
                        review.group_name
                    )
                )

            if metadata:
                st.caption(
                    " · ".join(metadata)
                )


def _display_rating_details(
    hotel: pd.Series,
):
    """Display detailed hotel rating criteria."""

    score_columns = st.columns(3)

    for index, (
        column,
        label,
    ) in enumerate(
        CRITERIA.items()
    ):
        value = hotel.get(
            column
        )

        score_columns[
            index % 3
        ].metric(
            label,
            (
                f"{_format_score(value)}/10"
                if pd.notna(value)
                else "N/A"
            ),
        )


def display_hotel_cards(
    recommendations: pd.DataFrame,
    hotel_info: pd.DataFrame,
    hotel_comments: pd.DataFrame,
    mode: str,
    score_label: str | None = None,
):
    """Display recommendation results as hotel cards."""

    hotel_catalog = (
        hotel_info
        .drop_duplicates(
            "hotel_id"
        )
        .set_index(
            "hotel_id",
            drop=False,
        )
    )

    for recommendation in (
        recommendations.itertuples(
            index=False
        )
    ):
        hotel_id = (
            recommendation.hotel_id
        )

        if (
            hotel_id
            not in hotel_catalog.index
        ):
            continue

        hotel = hotel_catalog.loc[
            hotel_id
        ]

        reviews = hotel_comments.loc[
            hotel_comments[
                "hotel_id"
            ].eq(hotel_id)
        ].copy()

        rank = getattr(
            recommendation,
            "rank",
            None,
        )

        safe_hotel_id = (
            str(hotel_id)
            .replace("_", "-")
        )

        if rank == 1:
            card_key = (
                f"hotel-card-top-"
                f"{safe_hotel_id}"
            )
        else:
            card_key = (
                f"hotel-card-"
                f"{safe_hotel_id}"
            )
        
        with st.container(
            border=False,
            key=card_key,
        ):
            hotel_name = str(
                hotel[
                    "hotel_name"
                ]
            )

            image_path, is_exact_image = (
                get_hotel_image(
                    hotel_id=hotel_id,
                    hotel_name=hotel_name,
                )
            )

            image_col, main_col, score_col = (
                st.columns(
                    [1, 4, 1],
                    gap="medium",
                )
            )

            # ================================================
            # Hotel image
            # ================================================

            with image_col:
                image_key = (
                    f"hotel-image-"
                    f"{safe_hotel_id}"
                )

                with st.container(
                    border=False,
                    key=image_key,
                ):
                    if image_path is not None:
                        st.image(
                            image_path,
                            width=260,
                        )

                        if not is_exact_image:
                            st.caption("Ảnh minh họa")

            with main_col:

                main_key = (
                    f"hotel-main-"
                    f"{safe_hotel_id}"
                )

                with st.container(
                    border=False,
                    key=main_key,
                ):

                    if rank is not None:
                        st.markdown(
                            f"### #{int(rank)} · "
                            f"{hotel_name}"
                        )
                    else:
                        st.markdown(
                            f"### {hotel_name}"
                        )

                    metadata_badges = []

                    hotel_rank = hotel.get(
                        "hotel_rank"
                    )

                    if pd.notna(hotel_rank):
                        metadata_badges.append(
                            (
                                "star",
                                "⭐",
                                (
                                    f"{float(hotel_rank):g} sao"
                                ),
                            )
                        )

                    total_score = hotel.get(
                        "total_score"
                    )

                    if pd.notna(total_score):
                        metadata_badges.append(
                            (
                                "score",
                                "🏅",
                                (
                                    f"{float(total_score):.1f}/10"
                                ),
                            )
                        )

                    comments_count = hotel.get(
                        "comments_count"
                    )

                    if pd.notna(comments_count):
                        metadata_badges.append(
                            (
                                "reviews",
                                "💬",
                                (
                                    f"{int(comments_count):,} "
                                    "reviews"
                                ),
                            )
                        )

                    badge_html = "".join(
                        (
                            '<span class="hotel-meta-badge '
                            f'{css_class}">'
                            f'{icon} {text}'
                            '</span>'
                        )
                        for (
                            css_class,
                            icon,
                            text,
                        )
                        in metadata_badges
                    )

                    if badge_html:
                        st.markdown(
                            (
                                '<div class="hotel-meta-row">'
                                f'{badge_html}'
                                '</div>'
                            ),
                            unsafe_allow_html=True,
                        )

                    highlights = (
                        extract_hotel_highlights(
                            hotel_name=hotel_name,
                            hotel_description=(
                                hotel.get(
                                    "hotel_description"
                                )
                            ),
                            hotel_address=(
                                hotel.get(
                                    "hotel_address"
                                )
                            ),
                            max_tags=5,
                        )
                    )

                    if highlights:
                        highlight_html = "".join(
                            (
                                '<span class="'
                                'hotel-highlight-tag">'
                                f'{highlight}'
                                '</span>'
                            )
                            for highlight
                            in highlights
                        )

                        st.markdown(
                            (
                                '<div class="'
                                'hotel-highlight-row">'
                                f'{highlight_html}'
                                '</div>'
                            ),
                            unsafe_allow_html=True,
                        )

                    address = hotel.get(
                        "hotel_address"
                    )

                    if pd.notna(address):
                        address = str(
                            address
                        ).strip()

                        maps_url = (
                            _build_maps_url(
                                hotel_name=(
                                    hotel_name
                                ),
                                address=address,
                            )
                        )

                        st.link_button(
                            f"📍 {address}",
                            maps_url,
                            use_container_width=True,
                        )

                    else:
                        st.caption(
                            "📍 Chưa có dữ liệu địa chỉ."
                        )

            # ================================================
            # Recommendation score
            # ================================================

            with score_col:

                score_box_key = (
                    f"score-box-{safe_hotel_id}"
                )

                with st.container(
                    border=False,
                    key=score_box_key,
                ):

                    if mode == "content":

                        similarity = getattr(
                            recommendation,
                            "similarity_score",
                            0,
                        )

                        similarity = float(
                            similarity
                        )

                        fit_percent = (
                            similarity * 100
                        )

                        fit_percent = max(
                            0.0,
                            min(
                                100.0,
                                fit_percent,
                            ),
                        )

                        st.metric(
                            (
                                score_label
                                or "Mức phù hợp"
                            ),
                            f"{fit_percent:.0f}%",
                        )

                        st.progress(
                            int(
                                round(
                                    fit_percent
                                )
                            )
                        )

                    elif (
                        mode
                        == "customer_group"
                    ):

                        prediction = getattr(
                            recommendation,
                            "prediction",
                            0,
                        )

                        prediction = float(
                            prediction
                        )

                        prediction_percent = (
                            prediction * 10
                        )

                        prediction_percent = max(
                            0.0,
                            min(
                                100.0,
                                prediction_percent,
                            ),
                        )

                        st.metric(
                            "Điểm dự đoán",
                            (
                                f"{prediction_percent:.0f}%"
                            ),
                        )

                        st.progress(
                            int(
                                round(
                                    prediction_percent
                                )
                            )
                        )

            # ================================================
            # Details and reviews
            # ================================================

            with st.expander(
                "Chi tiết và Đánh giá"
            ):
                st.markdown(
                    "#### Điểm đánh giá chi tiết"
                )

                _display_rating_details(
                    hotel
                )

                st.markdown(
                    "#### Mô tả khách sạn"
                )

                description = hotel.get(
                    "hotel_description"
                )

                if (
                    pd.notna(description)
                    and str(
                        description
                    ).strip()
                ):
                    st.write(
                        str(
                            description
                        ).strip()
                    )
                else:
                    st.info(
                        "Khách sạn chưa có "
                        "mô tả chi tiết."
                    )

                st.markdown(
                    "#### Đánh giá của khách hàng"
                )

                st.caption(
                    "Số review có trong dataset: "
                    f"{len(reviews):,}"
                )

                if reviews.empty:
                    st.info(
                        "Khách sạn này chưa có "
                        "review trong dataset."
                    )

                else:
                    highest_reviews = (
                        get_highest_reviews(
                            reviews,
                            top_n=5,
                        )
                    )

                    lowest_reviews = (
                        get_lowest_reviews(
                            reviews,
                            top_n=5,
                        )
                    )

                    high_tab, low_tab = (
                        st.tabs(
                            [
                                (
                                    "👍 5 review "
                                    "cao điểm nhất"
                                ),
                                (
                                    "👎 5 review "
                                    "thấp điểm nhất"
                                ),
                            ]
                        )
                    )

                    with high_tab:
                        _display_review_list(
                            highest_reviews
                        )

                    with low_tab:
                        _display_review_list(
                            lowest_reviews
                        )