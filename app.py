import pandas as pd
import streamlit as st

from src.insights import (
    build_benchmark_table,
    build_group_summary,
    build_keyword_tables,
    build_nationality_summary,
    build_overview_table,
    build_score_level_summary,
    build_yearly_summary,
    get_hotel_and_reviews,
    get_lowest_reviews,
)

from src.loaders import (
    load_cf_data,
    load_deployment_summary,
    load_hotel_comments,
    load_hotel_info,
    load_model_comparison,
    load_ridge_model,
    load_tfidf_matrix,
    load_tfidf_vectorizer,
)

from src.recommenders import (
    recommend_by_cosine,
    recommend_with_ridge,
    recommend_by_query,
)

from src.ui_components import (
    display_hotel_cards,
)

from src.ui_theme import (
    display_global_banner,
    load_app_styles,
)


st.set_page_config(
    page_title="Hotel Recommendation & Customer Insights",
    page_icon="🏨",
    layout="wide",
)

load_app_styles()

# ============================================================
# Load data and models
# ============================================================

try:
    hotel_info = load_hotel_info()
    hotel_comments = load_hotel_comments()
    cf_data = load_cf_data()
    model_comparison = load_model_comparison()
    summary = load_deployment_summary()
    tfidf_matrix = load_tfidf_matrix()
    ridge_model = load_ridge_model()
    tfidf_vectorizer = load_tfidf_vectorizer()

except Exception as error:
    st.error(
        "Không thể load dữ liệu hoặc model."
    )
    st.exception(error)
    st.stop()


# ============================================================
# Helper functions
# ============================================================

def create_hotel_options(
    catalog: pd.DataFrame,
) -> dict:
    catalog = (
        catalog
        .drop_duplicates("hotel_id")
        .sort_values("hotel_name")
    )

    return {
        (
            f"{row.hotel_name} "
            f"— ID: {row.hotel_id}"
        ): row.hotel_id
        for row in catalog.itertuples()
    }


PAGE_LABELS = {
    "Business Problem": (
        "🏠 Trang chủ"
    ),
    "Content-Based Recommendation": (
        "🔎 Tìm khách sạn"
    ),
    "Customer-Group Recommendation": (
        "👥 Gợi ý theo nhóm khách"
    ),
    "Hotel Insights": (
        "⭐ Khám phá & đánh giá"
    ),
    "Evaluation & Report": (
        "📊 Báo cáo mô hình"
    ),
    "Team Information": (
        "ℹ️ Về dự án"
    ),
}


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown(
    """
    ## 🏨 Hotel Explorer

    *Tìm nơi ở phù hợp với bạn*
    """
)

page = st.sidebar.radio(
    "Khám phá",
    list(
        PAGE_LABELS.keys()
    ),
    format_func=(
        lambda value:
        PAGE_LABELS[value]
    ),
)


# ============================================================
# Global banner
# ============================================================

display_global_banner()


# ============================================================
# Page 1: Business Problem
# ============================================================

if page == "Business Problem":
    st.header(
        "Hotel Explorer"
    )

    st.write(
        """
        Tìm khách sạn phù hợp với nhu cầu,
        khám phá gợi ý theo nhóm khách
        và xem insight từ đánh giá thực tế.
        """
    )

    st.write(
        """
        Ứng dụng hỗ trợ người dùng tìm khách sạn
        phù hợp và cung cấp insight cho chủ khách sạn.
        """
    )

    st.markdown(
        """
        **Ba nhiệm vụ chính**

        1. Gợi ý khách sạn có nội dung tương tự.
        2. Gợi ý khách sạn theo phân khúc khách hàng.
        3. Phân tích review và hiệu quả của từng khách sạn.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Khách sạn",
        f"{summary['hotel_info_rows']:,}",
    )

    col2.metric(
        "Review",
        f"{summary['hotel_comment_rows']:,}",
    )

    col3.metric(
        "Customer groups",
        f"{summary['customer_groups']:,}",
    )

    col4.metric(
        "Customer group – hotel interactions",
        f"{summary['cf_interactions']:,}",
    )

    st.subheader(
        "Thiết kế hệ thống"
    )

    st.code(
        """
Notebook:
Data preparation → Modeling → Evaluation → Export artifacts

Streamlit:
Load artifacts → User input → Recommendation / Insight
        """,
        language="text",
    )

    st.info(
        "Collaborative recommendation của project "
        "được thực hiện ở cấp customer group, "
        "không phải cá nhân hóa theo từng reviewer."
    )


# ============================================================
# Page 2: Content-Based Recommendation
# ============================================================

elif page == "Content-Based Recommendation":
    st.title(
        "🔎 Tìm khách sạn phù hợp"
    )

    st.write(
        """
        Tìm khách sạn theo nhu cầu của bạn
        hoặc chọn một khách sạn có sẵn để
        tìm những khách sạn tương tự.
        """
    )

    search_mode = st.radio(
        "Bạn muốn tìm theo cách nào?",
        [
            "Mô tả nhu cầu",
            "Khách sạn tương tự",
        ],
        horizontal=True,
    )

    top_n = st.slider(
        "Số khách sạn đề xuất",
        min_value=5,
        max_value=20,
        value=10,
        key="content_top_n",
    )

    # --------------------------------------------------------
    # Search by free-text query
    # --------------------------------------------------------

    if search_mode == "Mô tả nhu cầu":

        query = st.text_input(
            "Bạn đang tìm khách sạn như thế nào?",
            placeholder=(
                "Ví dụ: resort 5 sao gần biển "
                "ở Cam Ranh"
            ),
        )

        if st.button(
            "🔍 Tìm khách sạn",
            key="query_search_button",
        ):
            try:
                result = recommend_by_query(
                    query=query,
                    hotel_info=hotel_info,
                    tfidf_matrix=tfidf_matrix,
                    tfidf_vectorizer=(
                        tfidf_vectorizer
                    ),
                    top_n=top_n,
                )

                st.success(
                    f"Tìm thấy {len(result)} "
                    "khách sạn phù hợp."
                )

                st.caption(
                    "Nếu mô tả có loại hình, hạng sao "
                    "hoặc khu vực được nhận diện, hệ thống "
                    "sẽ dùng các điều kiện đó để lọc ứng viên, "
                    "sau đó xếp hạng bằng TF-IDF + "
                    "Cosine Similarity. Phần trăm hiển thị "
                    "là độ tương đồng nội dung, không phải "
                    "xác suất người dùng sẽ đặt phòng."
                )

                display_hotel_cards(
                    recommendations=result,
                    hotel_info=hotel_info,
                    hotel_comments=hotel_comments,
                    mode="content",
                    score_label=(
                        "Mức phù hợp với nhu cầu"
                    ),
                )

            except Exception as error:
                st.error(str(error))

    # --------------------------------------------------------
    # Search hotels similar to an existing hotel
    # --------------------------------------------------------

    else:
        content_options = (
            create_hotel_options(
                hotel_info
            )
        )

        selected_label = st.selectbox(
            "Chọn khách sạn",
            list(
                content_options.keys()
            ),
        )

        selected_hotel_id = (
            content_options[
                selected_label
            ]
        )

        if st.button(
            "🔍 Tìm khách sạn tương tự",
            key="similar_search_button",
        ):
            try:
                result = recommend_by_cosine(
                    hotel_id=(
                        selected_hotel_id
                    ),
                    hotel_info=hotel_info,
                    tfidf_matrix=tfidf_matrix,
                    top_n=top_n,
                )

                st.success(
                    f"Tìm thấy {len(result)} "
                    "khách sạn tương tự."
                )

                st.caption(
                    "Mức tương đồng được tính từ "
                    "Cosine Similarity × 100. "
                    "Đây là độ giống nhau về nội dung "
                    "giữa hai khách sạn."
                )

                display_hotel_cards(
                    recommendations=result,
                    hotel_info=hotel_info,
                    hotel_comments=hotel_comments,
                    mode="content",
                    score_label="Mức tương đồng",
                )

            except Exception as error:
                st.error(str(error))


# ============================================================
# Page 3: Customer-Group Recommendation
# ============================================================

elif page == "Customer-Group Recommendation":
    st.title(
        "👥 Customer-Group Recommendation"
    )

    st.warning(
        "Do mỗi Reviewer ID chỉ có một interaction, "
        "đây là recommendation theo phân khúc, "
        "không phải cá nhân hóa cho từng người."
    )

    group_lookup = (
        cf_data[["customer_group"]]
        .drop_duplicates()
        .copy()
    )

    group_lookup["nationality"] = (
        group_lookup["customer_group"]
        .map(
            lambda value: value.split(
                " | ",
                1,
            )[0]
        )
    )

    group_lookup["group_name"] = (
        group_lookup["customer_group"]
        .map(
            lambda value: value.split(
                " | ",
                1,
            )[1]
            if " | " in value
            else ""
        )
    )

    nationality = st.selectbox(
        "Quốc tịch",
        sorted(
            group_lookup[
                "nationality"
            ].unique()
        ),
    )

    valid_group_names = sorted(
        group_lookup.loc[
            group_lookup[
                "nationality"
            ].eq(nationality),
            "group_name",
        ].unique()
    )

    group_name = st.selectbox(
        "Nhóm khách",
        valid_group_names,
    )

    matching_group = group_lookup.loc[
        group_lookup[
            "nationality"
        ].eq(nationality)
        & group_lookup[
            "group_name"
        ].eq(group_name),
        "customer_group",
    ]

    customer_group = (
        matching_group.iloc[0]
    )

    top_n = st.slider(
        "Số khách sạn đề xuất",
        min_value=5,
        max_value=20,
        value=10,
        key="ridge_top_n",
    )

    if st.button(
        "Tạo recommendation",
        key="ridge_button",
    ):
        try:
            result = recommend_with_ridge(
                customer_group=customer_group,
                hotel_info=hotel_info,
                cf_data=cf_data,
                ridge_model=ridge_model,
                top_n=top_n,
            )

            st.success(
                f"Tìm thấy {len(result)} "
                "khách sạn phù hợp với nhóm: "
                f"{customer_group}"
            )

            st.caption(
                "Điểm dự đoán là rating mà mô hình "
                "ước lượng cho từng khách sạn đối với "
                "nhóm khách đã chọn, trên thang 0–10."
            )

            display_hotel_cards(
                recommendations=result,
                hotel_info=hotel_info,
                hotel_comments=hotel_comments,
                mode="customer_group",
                customer_group=customer_group,
            )

        except Exception as error:
            st.exception(error)


# ============================================================
# Page 4: Hotel Insights
# ============================================================

elif page == "Hotel Insights":
    st.title(
        "📊 Hotel Insights"
    )

    reviewed_ids = set(
        hotel_comments["hotel_id"]
        .dropna()
        .unique()
    )

    insight_catalog = hotel_info.loc[
        hotel_info["hotel_id"].isin(
            reviewed_ids
        )
    ].copy()

    insight_options = create_hotel_options(
        insight_catalog
    )

    selected_label = st.selectbox(
        "Chọn khách sạn cần phân tích",
        list(insight_options.keys()),
    )

    hotel_id = insight_options[
        selected_label
    ]

    try:
        hotel, reviews = get_hotel_and_reviews(
            hotel_id=hotel_id,
            hotel_info=hotel_info,
            hotel_comments=hotel_comments,
        )

        st.subheader(
            hotel["hotel_name"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Số review",
            f"{len(reviews):,}",
        )

        col2.metric(
            "Điểm trung bình",
            f"{reviews['score'].mean():.2f}",
        )

        col3.metric(
            "Hạng sao",
            (
                f"{hotel['hotel_rank']}"
                if pd.notna(
                    hotel["hotel_rank"]
                )
                else "Không có dữ liệu"
            ),
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Tổng quan",
                "Điểm mạnh",
                "Khách hàng",
                "Xu hướng",
                "Review & từ khóa",
            ]
        )

        with tab1:
            overview = build_overview_table(
                hotel,
                reviews,
            )

            st.dataframe(
                overview,
                use_container_width=True,
                hide_index=True,
            )

            score_levels = (
                build_score_level_summary(
                    reviews
                )
            )

            st.subheader(
                "Phân bố mức đánh giá"
            )

            st.dataframe(
                score_levels,
                use_container_width=True,
                hide_index=True,
            )

            if not score_levels.empty:
                st.bar_chart(
                    score_levels.set_index(
                        "score_level"
                    )["review_count"]
                )

        with tab2:
            benchmark = build_benchmark_table(
                hotel,
                hotel_info,
            )

            st.dataframe(
                benchmark.round(2),
                use_container_width=True,
                hide_index=True,
            )

            chart_data = (
                benchmark
                .dropna(
                    subset=[
                        "Điểm khách sạn",
                        "TB toàn hệ thống",
                    ]
                )
                .set_index("Tiêu chí")
                [
                    [
                        "Điểm khách sạn",
                        "TB toàn hệ thống",
                        "TB cùng hạng sao",
                    ]
                ]
            )

            if not chart_data.empty:
                st.bar_chart(chart_data)

            valid_benchmark = (
                benchmark
                .dropna(
                    subset=[
                        "Chênh lệch với hệ thống"
                    ]
                )
                .sort_values(
                    "Chênh lệch với hệ thống",
                    ascending=False,
                )
            )

            if not valid_benchmark.empty:
                best = valid_benchmark.iloc[0]
                lowest = valid_benchmark.iloc[-1]

                st.success(
                    "Tiêu chí nổi bật nhất: "
                    f"{best['Tiêu chí']} "
                    f"({best['Chênh lệch với hệ thống']:+.2f})"
                )

                if (
                    lowest[
                        "Chênh lệch với hệ thống"
                    ]
                    < 0
                ):
                    st.warning(
                        "Tiêu chí cần ưu tiên: "
                        f"{lowest['Tiêu chí']} "
                        f"({lowest['Chênh lệch với hệ thống']:+.2f})"
                    )
                else:
                    st.info(
                        "Không có tiêu chí có dữ liệu "
                        "thấp hơn trung bình hệ thống."
                    )

        with tab3:

            nationality_summary = build_nationality_summary(
                reviews,
                min_reviews=20,
            )

            group_summary = build_group_summary(
                reviews,
                min_reviews=20,
            )

            # Fallback cho khách sạn ít dữ liệu.
            if nationality_summary.empty:
                nationality_summary = build_nationality_summary(
                    reviews,
                    min_reviews=1,
                )

            if group_summary.empty:
                group_summary = build_group_summary(
                    reviews,
                    min_reviews=1,
                )

            if len(reviews) < 20:
                st.warning(
                    f"Khách sạn này chỉ có {len(reviews)} review. "
                    "Thống kê quốc tịch và nhóm khách chỉ mang tính mô tả, "
                    "chưa đủ dữ liệu để kết luận xu hướng."
                )

            left, right = st.columns(2)

            with left:
                st.subheader(
                    "Quốc tịch"
                )

                st.dataframe(
                    nationality_summary,
                    use_container_width=True,
                    hide_index=True,
                )

                if not nationality_summary.empty:
                    st.bar_chart(
                        nationality_summary
                        .set_index("nationality")
                        ["review_count"]
                    )

            with right:
                st.subheader(
                    "Nhóm khách"
                )

                st.dataframe(
                    group_summary,
                    use_container_width=True,
                    hide_index=True,
                )

                if not group_summary.empty:
                    st.bar_chart(
                        group_summary
                        .set_index("group_name")
                        ["review_count"]
                    )

        with tab4:
            yearly_summary = (
                build_yearly_summary(
                    reviews
                )
            )

            st.dataframe(
                yearly_summary,
                use_container_width=True,
                hide_index=True,
            )

            if not yearly_summary.empty:
                st.subheader(
                    "Số review theo năm"
                )

                st.bar_chart(
                    yearly_summary
                    .set_index("review_year")
                    ["review_count"]
                )

                st.subheader(
                    "Điểm trung bình theo năm"
                )

                st.line_chart(
                    yearly_summary
                    .set_index("review_year")
                    ["average_score"]
                )

        with tab5:
            (
                positive_keywords,
                improvement_keywords,
                keyword_counts,
            ) = build_keyword_tables(
                reviews=reviews,
                hotel_name=hotel[
                    "hotel_name"
                ],
            )

            left, right = st.columns(2)

            with left:
                st.subheader(
                    "Review điểm từ 9 trở lên"
                )

                st.metric(
                    "Số review",
                    keyword_counts[
                        "high_score_reviews"
                    ],
                )

                if positive_keywords.empty:
                    st.info(
                        "Chưa đủ từ khóa."
                    )
                else:
                    st.dataframe(
                        positive_keywords,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.bar_chart(
                        positive_keywords
                        .set_index("keyword")
                        ["frequency"]
                    )

            with right:
                st.subheader(
                    "Review điểm dưới 9"
                )

                st.metric(
                    "Số review",
                    keyword_counts[
                        "lower_score_reviews"
                    ],
                )

                if improvement_keywords.empty:
                    st.info(
                        "Chưa đủ từ khóa."
                    )
                else:
                    st.dataframe(
                        improvement_keywords,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.bar_chart(
                        improvement_keywords
                        .set_index("keyword")
                        ["frequency"]
                    )

            st.caption(
                "Từ khóa chỉ giúp xác định chủ đề "
                "cần đọc sâu hơn; không tự động kết luận "
                "đó là điểm mạnh hay điểm yếu."
            )

            st.subheader(
                "Các review có điểm thấp nhất"
            )

            lowest_reviews = get_lowest_reviews(
                reviews,
                top_n=10,
            )

            st.dataframe(
                lowest_reviews,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:
        st.exception(error)


# ============================================================
# Page 5: Evaluation
# ============================================================

elif page == "Evaluation & Report":
    st.title(
        "📈 Evaluation & Report"
    )

    st.subheader(
        "Content-Based Filtering"
    )

    st.metric(
        "Top-10 overlap: Cosine và Gensim",
        (
            f"{summary['content_top10_overlap']}/10"
        ),
    )

    st.write(
        """
        Hai phương pháp có mức nhất quán tương đối,
        nhưng similarity score không phải xác suất
        người dùng sẽ đặt khách sạn.
        """
    )

    st.info(
        "Chưa thể tính Precision@K hoặc Recall@K "
        "do dữ liệu không có ground truth recommendation."
    )

    st.subheader(
        "Collaborative Filtering"
    )

    sorted_comparison = (
        model_comparison
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    st.dataframe(
        sorted_comparison.round(4),
        use_container_width=True,
        hide_index=True,
    )

    metric_chart = (
        model_comparison
        .set_index("model")
        [
            [
                "RMSE",
                "MAE",
            ]
        ]
    )

    st.bar_chart(metric_chart)

    st.success(
        "One-Hot + Ridge được dùng trong GUI "
        "vì có RMSE và MAE thấp nhất trên tập test."
    )

    st.write(
        """
        PySpark ALS vẫn được xây dựng và đánh giá
        trong notebook để đáp ứng yêu cầu Big Data
        in Machine Learning, nhưng không chạy trực tiếp
        trong Streamlit.
        """
    )

    st.subheader(
        "Limitations"
    )

    for limitation in summary["limitations"]:
        st.markdown(f"- {limitation}")


# ============================================================
# Page 6: Team
# ============================================================

elif page == "Team Information":
    st.title(
        "👨‍💻 Team Information"
    )

    st.subheader(
        "Nhóm 2"
    )

    st.write(
        """
        - Nguyễn Minh Khoa
        - Nguyễn Hoàng Quỳnh Anh
        """
    )

    st.subheader(
        "Project Link"
    )

    st.write(
        """
        - Streamlit: https://hotel-recommendation-insights.streamlit.app/
        """
    )