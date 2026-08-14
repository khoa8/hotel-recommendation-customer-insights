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
    RecommendationError,
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

from src.i18n import (
    LANGUAGE_OPTIONS,
    localize_dataframe,
    localize_value,
    t,
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
        "Error in loading data/models."
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


PAGE_LABEL_KEYS = {
    "Business Problem": (
        "nav.home"
    ),
    "Content-Based Recommendation": (
        "nav.search"
    ),
    "Customer-Group Recommendation": (
        "nav.group"
    ),
    "Hotel Insights": (
        "nav.insights"
    ),
    "Evaluation & Report": (
        "nav.evaluation"
    ),
}


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown(
    "## 🏨 Hotel Explorer"
)

language = st.sidebar.radio(
    "Language/Ngôn ngữ",
    list(
        LANGUAGE_OPTIONS.keys()
    ),
    index=0,
    format_func=(
        lambda code:
        LANGUAGE_OPTIONS[
            code
        ]
    ),
    horizontal=True,
    key="language_code",
)

st.sidebar.caption(
    t(
        "sidebar.tagline",
        language,
    )
)

page = st.sidebar.radio(
    t(
        "sidebar.explore",
        language,
    ),
    list(
        PAGE_LABEL_KEYS.keys()
    ),
    format_func=(
        lambda value:
        t(
            PAGE_LABEL_KEYS[
                value
            ],
            language,
        )
    ),
)


# ============================================================
# Global banner
# ============================================================

display_global_banner(language)


# ============================================================
# Page 1: Business Problem
# ============================================================

if page == "Business Problem":
    st.header(
        t(
            "home.title",
            language,
        )
    )

    st.write(
        t(
            "home.intro_1",
            language,
        )
    )

    st.write(
        t(
            "home.intro_2",
            language,
        )
    )

    st.markdown(
        (
            f"**{t('home.tasks_title', language)}**\n\n"
            f"1. {t('home.task_1', language)}\n"
            f"2. {t('home.task_2', language)}\n"
            f"3. {t('home.task_3', language)}"
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        t(
            "home.metric_hotels",
            language,
        ),
        f"{summary['hotel_info_rows']:,}",
    )

    col2.metric(
        t(
            "home.metric_reviews",
            language,
        ),
        f"{summary['hotel_comment_rows']:,}",
    )

    col3.metric(
        t(
            "home.metric_groups",
            language,
        ),
        f"{summary['customer_groups']:,}",
    )

    col4.metric(
        t(
            "home.metric_interactions",
            language,
        ),
        f"{summary['cf_interactions']:,}",
    )

    st.subheader(
        t(
            "home.system_design",
            language,
        )
    )

    st.code(
        t(
            "home.system_flow",
            language,
        ),
        language="text",
    )

    st.info(
        t(
            "home.group_note",
            language,
        )
    )


# ============================================================
# Page 2: Content-Based Recommendation
# ============================================================

elif page == "Content-Based Recommendation":
    st.title(
        t(
            "search.title",
            language,
        )
    )

    st.write(
        t(
            "search.intro",
            language,
        )
    )

    search_mode = st.radio(
        t(
            "search.mode_question",
            language,
        ),
        [
            "query",
            "similar",
        ],
        format_func=(
            lambda mode:
            t(
                (
                    "search.mode_query"
                    if mode == "query"
                    else "search.mode_similar"
                ),
                language,
            )
        ),
        horizontal=True,
    )

    top_n = st.slider(
        t(
            "search.top_n",
            language,
        ),
        min_value=5,
        max_value=20,
        value=10,
        key="content_top_n",
    )

    # --------------------------------------------------------
    # Search by free-text query
    # --------------------------------------------------------

    if search_mode == "query":

        query = st.text_input(
            t(
                "search.query_label",
                language,
            ),
            placeholder=(
                t(
                    "search.query_placeholder",
                    language,
                )
            ),
        )

        if st.button(
            t(
                "search.button",
                language,
            ),
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
                    t(
                        "search.found",
                        language,
                        count=len(result),
                    )
                )

                st.caption(
                    t(
                        "search.ranking_explanation",
                        language,
                    )
                )

                display_hotel_cards(
                    recommendations=result,
                    hotel_info=hotel_info,
                    hotel_comments=hotel_comments,
                    mode="content",
                    language=language,
                    score_label_key=(
                        "card.match_score"
                    ),
                )

            except RecommendationError as error:
                st.error(
                    t(
                        error.message_key,
                        language,
                    )
                )

            except Exception as error:
                st.exception(
                    error
                )

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
            t(
                "search.choose_hotel",
                language,
            ),
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
            t(
                "search.similar_button",
                language,
            ),
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
                    t(
                        "search.similar_found",
                        language,
                        count=len(result),
                    )
                )

                st.caption(
                    t(
                        "search.similarity_explanation",
                        language,
                    )
                )

                display_hotel_cards(
                    recommendations=result,
                    hotel_info=hotel_info,
                    hotel_comments=hotel_comments,
                    mode="content",
                    language=language,
                    score_label_key=(
                        "card.similarity_score"
                    ),
                )

            except RecommendationError as error:
                st.error(
                    t(
                        error.message_key,
                        language,
                    )
                )

            except Exception as error:
                st.exception(
                    error
                )


# ============================================================
# Page 3: Customer-Group Recommendation
# ============================================================

elif page == "Customer-Group Recommendation":
    st.title(
        t(
            "group.title",
            language,
        )
    )

    st.warning(
        t(
            "group.warning",
            language,
        )
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
        t(
            "group.nationality",
            language,
        ),
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
        t(
            "group.group_name",
            language,
        ),
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
        t(
            "group.top_n",
            language,
        ),
        min_value=5,
        max_value=20,
        value=10,
        key="ridge_top_n",
    )

    if st.button(
        t(
            "group.button",
            language,
        ),
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
                t(
                    "group.found",
                    language,
                    count=len(result),
                    group=customer_group,
                )
            )

            st.caption(
                t(
                    "group.explanation",
                    language,
                )
            )

            display_hotel_cards(
                recommendations=result,
                hotel_info=hotel_info,
                hotel_comments=hotel_comments,
                mode="customer_group",
                language=language,
            )

        except RecommendationError as error:
            st.error(
                t(
                    error.message_key,
                    language,
                )
            )

        except Exception as error:
            st.exception(
                error
            )


# ============================================================
# Page 4: Hotel Insights
# ============================================================

elif page == "Hotel Insights":
    st.title(
        t(
            "insights.title",
            language,
        )
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
        t(
            "insights.choose_hotel",
            language,
        ),
        list(
            insight_options.keys()
        ),
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
            t(
                "insights.review_count",
                language,
            ),
            f"{len(reviews):,}",
        )

        col2.metric(
            t(
                "insights.average_score",
                language,
            ),
            f"{reviews['score'].mean():.2f}",
        )

        col3.metric(
            t(
                "insights.star_rating",
                language,
            ),
            (
                f"{hotel['hotel_rank']}"
                if pd.notna(
                    hotel["hotel_rank"]
                )
                else t(
                    "common.no_data",
                    language,
                )
            ),
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                t(
                    "insights.tab_overview",
                    language,
                ),
                t(
                    "insights.tab_strengths",
                    language,
                ),
                t(
                    "insights.tab_customers",
                    language,
                ),
                t(
                    "insights.tab_trends",
                    language,
                ),
                t(
                    "insights.tab_keywords",
                    language,
                ),
            ]
        )

        with tab1:
            overview = build_overview_table(
                hotel,
                reviews,
            )

            st.dataframe(
                localize_dataframe(
                    overview,
                    language,
                    translate_values=True,
                ),
                use_container_width=True,
                hide_index=True,
            )

            score_levels = (
                build_score_level_summary(
                    reviews
                )
            )

            st.subheader(
                t(
                    "insights.score_distribution",
                    language,
                )
            )

            st.dataframe(
                localize_dataframe(
                    score_levels,
                    language,
                ),
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
                localize_dataframe(
                    benchmark.round(2),
                    language,
                    translate_values=True,
                ),
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
                    t(
                        "insights.best_criterion",
                        language,
                        criterion=(
                            localize_value(
                                best[
                                    "Tiêu chí"
                                ],
                                language,
                            )
                        ),
                        value=(
                            best[
                                "Chênh lệch với hệ thống"
                            ]
                        ),
                    )
                )

                if (
                    lowest[
                        "Chênh lệch với hệ thống"
                    ]
                    < 0
                ):
                    st.warning(
                        t(
                            "insights.priority_criterion",
                            language,
                            criterion=(
                                localize_value(
                                    lowest[
                                        "Tiêu chí"
                                    ],
                                    language,
                                )
                            ),
                            value=(
                                lowest[
                                    "Chênh lệch với hệ thống"
                                ]
                            ),
                        )
                    )
                else:
                    st.info(
                        t(
                            "insights.no_below_average",
                            language,
                        )
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
                    t(
                        "insights.low_data_warning",
                        language,
                        count=len(reviews)
                    )
                )

            left, right = st.columns(2)

            with left:
                st.subheader(
                    t(
                        "insights.nationality",
                        language,
                    )
                )

                st.dataframe(
                    localize_dataframe(
                        nationality_summary,
                        language,
                    ),
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
                    t(
                        "insights.group",
                        language,
                    )
                )

                st.dataframe(
                    localize_dataframe(
                        group_summary,
                        language,
                    ),
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
                localize_dataframe(
                    yearly_summary,
                    language,
                ),
                use_container_width=True,
                hide_index=True,
            )

            if not yearly_summary.empty:
                st.subheader(
                    t(
                        "insights.reviews_by_year",
                        language,
                    )
                )

                st.bar_chart(
                    yearly_summary
                    .set_index("review_year")
                    ["review_count"]
                )

                st.subheader(
                    t(
                        "insights.score_by_year",
                        language,
                    )
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
                    t(
                        "insights.high_reviews",
                        language,
                    )
                )

                st.metric(
                    "Số review",
                    keyword_counts[
                        "high_score_reviews"
                    ],
                )

                if positive_keywords.empty:
                    st.info(
                        t(
                            "insights.no_keywords",
                            language,
                        )
                    )
                else:
                    st.dataframe(
                        localize_dataframe(
                            positive_keywords,
                            language,
                        ),
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
                    t(
                        "insights.low_reviews",
                        language,
                    )
                )

                st.metric(
                    "Số review",
                    keyword_counts[
                        "lower_score_reviews"
                    ],
                )

                if improvement_keywords.empty:
                    st.info(
                        t(
                             "insights.no_keywords",
                             language,
                        )
                    )
                else:
                    st.dataframe(
                        localize_dataframe(
                            improvement_keywords,
                            language,
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.bar_chart(
                        improvement_keywords
                        .set_index("keyword")
                        ["frequency"]
                    )

            st.caption(
                t(
                    "insights.keyword_note",
                    language,
                )
            )

            st.subheader(
               t(
                    "insights.lowest_reviews",
                    language,
               )
            )

            lowest_reviews = get_lowest_reviews(
                reviews,
                top_n=10,
            )

            st.dataframe(
                localize_dataframe(
                    lowest_reviews,
                    language,
                ),
                use_container_width=True,
                hide_index=True,
            )

    except RecommendationError as error:
        st.error(
            t(
                error.message_key,
                language,
            )
        )

    except Exception as error:
        st.exception(
            error
        )


# ============================================================
# Page 5: Evaluation
# ============================================================

elif page == "Evaluation & Report":
    st.title(
        t(
            "evaluation.title",
            language,
        )
    )

    st.subheader(
        t(
            "evaluation.content_based",
            language,
        )
    )

    st.metric(
        t(
            "evaluation.overlap",
            language,
        ),
        (
            f"{summary['content_top10_overlap']}/10"
        ),
    )

    st.write(
        t(
            "evaluation.content_note",
            language,
        )
    )

    st.info(
        t(
            "evaluation.no_ground_truth",
            language,
        )
    )

    st.subheader(
        t(
            "evaluation.collaborative",
            language,
        )
    )

    sorted_comparison = (
        model_comparison
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    st.dataframe(
        localize_dataframe(
            sorted_comparison.round(4),
            language,
        ),
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
        t(
            "evaluation.ridge_selected",
            language,
        )
    )

    st.write(
        t(
            "evaluation.als_note",
            language,
        )
    )

    st.subheader(
        t(
            "evaluation.limitations",
            language,
        )
    )

    LIMITATION_KEYS = [
        "limitation.single_interaction",
        "limitation.group_level",
        "limitation.hotel_id",
        "limitation.no_ground_truth",
    ]

    for limitation_key in (
        LIMITATION_KEYS
    ):
        st.markdown(
            "- "
            + t(
                limitation_key,
                language,
            )
        )
