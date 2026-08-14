import pandas as pd


# ============================================================
# Languages
# ============================================================

LANGUAGE_OPTIONS = {
    "vi": "🇻🇳 Tiếng Việt",
    "en": "🇬🇧 English",
}


# ============================================================
# UI translations
# ============================================================

TRANSLATIONS = {
    "vi": {
        # ----------------------------------------------------
        # Sidebar / navigation
        # ----------------------------------------------------
        "sidebar.tagline": (
            "Tìm nơi ở phù hợp với bạn"
        ),
        "sidebar.explore": "Khám phá",

        "nav.home": "🏠 Trang chủ",
        "nav.search": "🔎 Tìm khách sạn",
        "nav.group": "👥 Gợi ý theo nhóm khách",
        "nav.insights": "⭐ Khám phá & đánh giá",
        "nav.evaluation": "📊 Báo cáo mô hình",

        # ----------------------------------------------------
        # Global hero
        # ----------------------------------------------------
        "hero.title": (
            "Tìm nơi ở phù hợp cho chuyến đi của bạn"
        ),

        # ----------------------------------------------------
        # Home
        # ----------------------------------------------------
        "home.title": "Hotel Explorer",
        "home.intro_1": (
            "Tìm khách sạn phù hợp với nhu cầu, "
            "khám phá gợi ý theo nhóm khách "
            "và xem insight từ đánh giá thực tế."
        ),
        "home.intro_2": (
            "Ứng dụng hỗ trợ người dùng tìm khách sạn "
            "phù hợp và cung cấp insight cho chủ khách sạn."
        ),
        "home.tasks_title": "Ba nhiệm vụ chính",
        "home.task_1": (
            "Gợi ý khách sạn có nội dung tương tự."
        ),
        "home.task_2": (
            "Gợi ý khách sạn theo phân khúc khách hàng."
        ),
        "home.task_3": (
            "Phân tích review và hiệu quả "
            "của từng khách sạn."
        ),
        "home.metric_hotels": "Khách sạn",
        "home.metric_reviews": "Review",
        "home.metric_groups": "Nhóm khách hàng",
        "home.metric_interactions": (
            "Tương tác nhóm khách – khách sạn"
        ),
        "home.system_design": "Thiết kế hệ thống",
        "home.system_flow": (
            "Notebook:\n"
            "Data preparation → Modeling → "
            "Evaluation → Export artifacts\n\n"
            "Streamlit:\n"
            "Load artifacts → User input → "
            "Recommendation / Insight"
        ),
        "home.group_note": (
            "Collaborative recommendation của project "
            "được thực hiện ở cấp customer group, "
            "không phải cá nhân hóa theo từng reviewer."
        ),

        # ----------------------------------------------------
        # Content-based search
        # ----------------------------------------------------
        "search.title": "🔎 Tìm khách sạn phù hợp",
        "search.intro": (
            "Tìm khách sạn theo nhu cầu của bạn "
            "hoặc chọn một khách sạn có sẵn để "
            "tìm những khách sạn tương tự."
        ),
        "search.mode_question": (
            "Bạn muốn tìm theo cách nào?"
        ),
        "search.mode_query": "Mô tả nhu cầu",
        "search.mode_similar": "Khách sạn tương tự",
        "search.top_n": "Số khách sạn đề xuất",
        "search.query_label": (
            "Bạn đang tìm khách sạn như thế nào?"
        ),
        "search.query_placeholder": (
            "Ví dụ: resort 5 sao gần biển ở Cam Ranh"
        ),
        "search.button": "🔍 Tìm khách sạn",
        "search.found": (
            "Tìm thấy {count} khách sạn phù hợp."
        ),
        "search.ranking_explanation": (
            "Nếu mô tả có loại hình, hạng sao hoặc khu vực "
            "được nhận diện, hệ thống sẽ dùng các điều kiện "
            "đó để lọc ứng viên, sau đó xếp hạng bằng "
            "TF-IDF + Cosine Similarity. Phần trăm hiển thị "
            "là độ tương đồng nội dung, không phải xác suất "
            "người dùng sẽ đặt phòng."
        ),
        "search.choose_hotel": "Chọn khách sạn",
        "search.similar_button": (
            "🔍 Tìm khách sạn tương tự"
        ),
        "search.similar_found": (
            "Tìm thấy {count} khách sạn tương tự."
        ),
        "search.similarity_explanation": (
            "Mức tương đồng được tính từ "
            "Cosine Similarity × 100. "
            "Đây là độ giống nhau về nội dung "
            "giữa hai khách sạn."
        ),

        # ----------------------------------------------------
        # Customer group
        # ----------------------------------------------------
        "group.title": (
            "👥 Gợi ý theo nhóm khách"
        ),
        "group.warning": (
            "Do mỗi Reviewer ID chỉ có một interaction, "
            "đây là recommendation theo phân khúc, "
            "không phải cá nhân hóa cho từng người."
        ),
        "group.nationality": "Quốc tịch",
        "group.group_name": "Nhóm khách",
        "group.top_n": "Số khách sạn đề xuất",
        "group.button": "Tạo gợi ý",
        "group.found": (
            "Tìm thấy {count} khách sạn phù hợp "
            "với nhóm: {group}"
        ),
        "group.explanation": (
            "Điểm dự đoán là rating mà mô hình "
            "ước lượng cho từng khách sạn đối với "
            "nhóm khách đã chọn. "
            "Đây không phải xác suất đặt phòng."
        ),

        # ----------------------------------------------------
        # Hotel insights
        # ----------------------------------------------------
        "insights.title": "📊 Khám phá & đánh giá",
        "insights.choose_hotel": (
            "Chọn khách sạn cần phân tích"
        ),
        "insights.review_count": "Số review",
        "insights.average_score": "Điểm trung bình",
        "insights.star_rating": "Hạng sao",
        "common.no_data": "Không có dữ liệu",

        "insights.tab_overview": "Tổng quan",
        "insights.tab_strengths": "Điểm mạnh",
        "insights.tab_customers": "Khách hàng",
        "insights.tab_trends": "Xu hướng",
        "insights.tab_keywords": "Review & từ khóa",

        "insights.score_distribution": (
            "Phân bố mức đánh giá"
        ),
        "insights.best_criterion": (
            "Tiêu chí nổi bật nhất: {criterion} ({value:+.2f})"
        ),
        "insights.priority_criterion": (
            "Tiêu chí cần ưu tiên: "
            "{criterion} ({value:+.2f})"
        ),
        "insights.no_below_average": (
            "Không có tiêu chí có dữ liệu "
            "thấp hơn trung bình hệ thống."
        ),
        "insights.low_data_warning": (
            "Khách sạn này chỉ có {count} review. "
            "Thống kê quốc tịch và nhóm khách "
            "chỉ mang tính mô tả, chưa đủ dữ liệu "
            "để kết luận xu hướng."
        ),
        "insights.nationality": "Quốc tịch",
        "insights.group": "Nhóm khách",
        "insights.reviews_by_year": (
            "Số review theo năm"
        ),
        "insights.score_by_year": (
            "Điểm trung bình theo năm"
        ),
        "insights.high_reviews": (
            "Review điểm từ 9 trở lên"
        ),
        "insights.low_reviews": (
            "Review điểm dưới 9"
        ),
        "insights.no_keywords": (
            "Chưa đủ từ khóa."
        ),
        "insights.keyword_note": (
            "Từ khóa chỉ giúp xác định chủ đề "
            "cần đọc sâu hơn; không tự động kết luận "
            "đó là điểm mạnh hay điểm yếu."
        ),
        "insights.lowest_reviews": (
            "Các review có điểm thấp nhất"
        ),

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------
        "evaluation.title": "📈 Báo cáo mô hình",
        "evaluation.content_based": (
            "Content-Based Filtering"
        ),
        "evaluation.overlap": (
            "Top-10 overlap: Cosine và Gensim"
        ),
        "evaluation.content_note": (
            "Hai phương pháp có mức nhất quán tương đối, "
            "nhưng similarity score không phải xác suất "
            "người dùng sẽ đặt khách sạn."
        ),
        "evaluation.no_ground_truth": (
            "Chưa thể tính Precision@K hoặc Recall@K "
            "do dữ liệu không có ground truth recommendation."
        ),
        "evaluation.collaborative": (
            "Collaborative Filtering"
        ),
        "evaluation.ridge_selected": (
            "One-Hot + Ridge được dùng trong GUI "
            "vì có RMSE và MAE thấp nhất trên tập test."
        ),
        "evaluation.als_note": (
            "PySpark ALS vẫn được xây dựng và đánh giá "
            "trong notebook để đáp ứng yêu cầu Big Data "
            "in Machine Learning, nhưng không chạy trực tiếp "
            "trong Streamlit."
        ),
        "evaluation.limitations": "Hạn chế",

        "limitation.single_interaction": (
            "Mỗi Reviewer ID chỉ có một interaction."
        ),
        "limitation.group_level": (
            "Collaborative recommendation ở cấp "
            "customer group, không phải từng cá nhân."
        ),
        "limitation.hotel_id": (
            "Một số Hotel ID không khớp giữa hai bảng."
        ),
        "limitation.no_ground_truth": (
            "Chưa có ground truth recommendation để tính "
            "Precision@K hoặc Recall@K."
        ),

        # ----------------------------------------------------
        # Hotel cards
        # ----------------------------------------------------
        "card.image_caption": "Ảnh minh họa",
        "card.stars": "{value} sao",
        "card.review_count": "{count} reviews",
        "card.no_address": (
            "📍 Chưa có dữ liệu địa chỉ."
        ),
        "card.match_score": "Mức phù hợp",
        "card.similarity_score": "Mức tương đồng",
        "card.predicted_score": "Điểm dự đoán",
        "card.details": "Chi tiết và Đánh giá",
        "card.rating_details": (
            "Điểm đánh giá chi tiết"
        ),
        "card.description": "Mô tả khách sạn",
        "card.no_description": (
            "Khách sạn chưa có mô tả chi tiết."
        ),
        "card.customer_reviews": (
            "Đánh giá của khách hàng"
        ),
        "card.dataset_reviews": (
            "Số review: {count}"
        ),
        "card.no_reviews": (
            "Khách sạn này chưa có review trong dữ liệu."
        ),
        "card.high_reviews": (
            "👍 5 review cao điểm nhất"
        ),
        "card.low_reviews": (
            "👎 5 review thấp điểm nhất"
        ),
        "card.no_matching_reviews": (
            "Không có review phù hợp trong dữ liệu."
        ),

        # ----------------------------------------------------
        # Rating criteria
        # ----------------------------------------------------
        "criteria.location": "Vị trí",
        "criteria.cleanliness": "Sạch sẽ",
        "criteria.service": "Dịch vụ",
        "criteria.facilities": "Tiện nghi",
        "criteria.value_for_money": "Đáng giá tiền",
        "criteria.comfort_and_room_quality": (
            "Thoải mái & chất lượng phòng"
        ),

        # ----------------------------------------------------
        # Hotel highlights
        # ----------------------------------------------------
        "highlight.apartment": "🏠 Căn hộ",
        "highlight.villa": "🏡 Villa",
        "highlight.homestay": "🛏️ Homestay",
        "highlight.house": "🏘️ Nhà riêng",
        "highlight.resort": "🌴 Resort",
        "highlight.near_beach": "🏖️ Gần biển",
        "highlight.near_airport": "✈️ Gần sân bay",
        "highlight.airport_transfer": (
            "✈️ Đưa đón sân bay"
        ),
        "highlight.pool": "🏊 Hồ bơi",
        "highlight.spa": "💆 Spa",
        "highlight.gym": "🏋️ Phòng gym",
        "highlight.restaurant": "🍽️ Nhà hàng",
        "highlight.family": (
            "👨‍👩‍👧 Phù hợp gia đình"
        ),
        "highlight.luxury": "✨ Cao cấp",

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------
        "error.hotel_not_found": (
            "Không tìm thấy khách sạn."
        ),

        "error.query_required": (
            "Vui lòng nhập nhu cầu tìm khách sạn."
        ),

        "error.no_matching_keywords": (
            "Không tìm thấy từ khóa phù hợp "
            "trong dữ liệu khách sạn. "
            "Hãy thử mô tả khác."
        ),

        "error.no_candidates": (
            "Không có khách sạn thỏa đồng thời "
            "các điều kiện đã nhập. "
            "Hãy thử bớt một điều kiện."
        ),

        "error.no_content_match": (
            "Không tìm thấy khách sạn có "
            "nội dung phù hợp với mô tả."
        ),

        "error.no_gensim_recommendation": (
            "Không có Gensim recommendation "
            "cho khách sạn này."
        ),

        "error.customer_group_not_found": (
            "Không tìm thấy customer group."
        ),
    },

    "en": {
        # ----------------------------------------------------
        # Sidebar / navigation
        # ----------------------------------------------------
        "sidebar.tagline": (
            "Find the right place to stay"
        ),
        "sidebar.explore": "Explore",

        "nav.home": "🏠 Home",
        "nav.search": "🔎 Find Hotels",
        "nav.group": "👥 Group Recommendations",
        "nav.insights": "⭐ Hotel Insights",
        "nav.evaluation": "📊 Model Report",

        # ----------------------------------------------------
        # Global hero
        # ----------------------------------------------------
        "hero.title": (
            "Find the right stay for your trip"
        ),

        # ----------------------------------------------------
        # Home
        # ----------------------------------------------------
        "home.title": "Hotel Explorer",
        "home.intro_1": (
            "Find hotels that match your needs, "
            "explore recommendations by customer segment, "
            "and discover insights from real reviews."
        ),
        "home.intro_2": (
            "The application supports hotel discovery "
            "for travelers and provides customer insights "
            "for hotel analysis."
        ),
        "home.tasks_title": "Three main tasks",
        "home.task_1": (
            "Recommend hotels with similar content."
        ),
        "home.task_2": (
            "Recommend hotels by customer segment."
        ),
        "home.task_3": (
            "Analyze reviews and hotel performance."
        ),
        "home.metric_hotels": "Hotels",
        "home.metric_reviews": "Reviews",
        "home.metric_groups": "Customer groups",
        "home.metric_interactions": (
            "Customer group – hotel interactions"
        ),
        "home.system_design": "System design",
        "home.system_flow": (
            "Notebook:\n"
            "Data preparation → Modeling → "
            "Evaluation → Export artifacts\n\n"
            "Streamlit:\n"
            "Load artifacts → User input → "
            "Recommendation / Insight"
        ),
        "home.group_note": (
            "Collaborative recommendation in this project "
            "operates at customer-group level rather than "
            "personalizing recommendations for each reviewer."
        ),

        # ----------------------------------------------------
        # Content-based search
        # ----------------------------------------------------
        "search.title": "🔎 Find Hotels",
        "search.intro": (
            "Describe what you are looking for, "
            "or choose an existing hotel to find "
            "similar properties."
        ),
        "search.mode_question": (
            "How would you like to search?"
        ),
        "search.mode_query": "Describe your needs",
        "search.mode_similar": "Similar hotels",
        "search.top_n": "Number of recommendations",
        "search.query_label": (
            "What kind of hotel are you looking for?"
        ),
        "search.query_placeholder": (
            "Example: 5-star resort near the beach "
            "in Cam Ranh"
        ),
        "search.button": "🔍 Find Hotels",
        "search.found": (
            "Found {count} matching hotels."
        ),
        "search.ranking_explanation": (
            "When property type, star rating, or location "
            "can be identified, the system first filters "
            "candidate hotels and then ranks them using "
            "TF-IDF + Cosine Similarity. The displayed "
            "percentage represents content similarity, "
            "not the probability of booking."
        ),
        "search.choose_hotel": "Choose a hotel",
        "search.similar_button": (
            "🔍 Find Similar Hotels"
        ),
        "search.similar_found": (
            "Found {count} similar hotels."
        ),
        "search.similarity_explanation": (
            "Similarity is calculated as "
            "Cosine Similarity × 100. "
            "It represents content similarity "
            "between two hotels."
        ),

        # ----------------------------------------------------
        # Customer group
        # ----------------------------------------------------
        "group.title": (
            "👥 Group Recommendations"
        ),
        "group.warning": (
            "Because each Reviewer ID has only one "
            "interaction, recommendations are generated "
            "for customer segments rather than individuals."
        ),
        "group.nationality": "Nationality",
        "group.group_name": "Travel group",
        "group.top_n": "Number of recommendations",
        "group.button": "Generate Recommendations",
        "group.found": (
            "Found {count} hotels for group: {group}"
        ),
        "group.explanation": (
            "The predicted score is derived from the rating "
            "estimated by the model for each hotel and the "
            "selected customer group. "
            "It is not a booking probability."
        ),

        # ----------------------------------------------------
        # Hotel insights
        # ----------------------------------------------------
        "insights.title": "📊 Hotel Insights",
        "insights.choose_hotel": (
            "Choose a hotel to analyze"
        ),
        "insights.review_count": "Reviews",
        "insights.average_score": "Average score",
        "insights.star_rating": "Star rating",
        "common.no_data": "No data",

        "insights.tab_overview": "Overview",
        "insights.tab_strengths": "Strengths",
        "insights.tab_customers": "Customers",
        "insights.tab_trends": "Trends",
        "insights.tab_keywords": "Reviews & Keywords",

        "insights.score_distribution": (
            "Review score distribution"
        ),
        "insights.best_criterion": (
            "Strongest criterion: "
            "{criterion} ({value:+.2f})"
        ),
        "insights.priority_criterion": (
            "Priority for improvement: "
            "{criterion} ({value:+.2f})"
        ),
        "insights.no_below_average": (
            "No criterion with available data "
            "is below the system average."
        ),
        "insights.low_data_warning": (
            "This hotel has only {count} reviews. "
            "Nationality and traveler-group statistics "
            "are descriptive and are not sufficient "
            "to establish a reliable trend."
        ),
        "insights.nationality": "Nationality",
        "insights.group": "Travel group",
        "insights.reviews_by_year": "Reviews by year",
        "insights.score_by_year": (
            "Average score by year"
        ),
        "insights.high_reviews": (
            "Reviews scoring 9 or above"
        ),
        "insights.low_reviews": (
            "Reviews scoring below 9"
        ),
        "insights.no_keywords": (
            "Not enough keywords."
        ),
        "insights.keyword_note": (
            "Keywords identify themes that may deserve "
            "deeper review; they do not automatically "
            "prove a strength or weakness."
        ),
        "insights.lowest_reviews": (
            "Lowest-scoring reviews"
        ),

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------
        "evaluation.title": "📈 Model Evaluation",
        "evaluation.content_based": (
            "Content-Based Filtering"
        ),
        "evaluation.overlap": (
            "Top-10 overlap: Cosine vs. Gensim"
        ),
        "evaluation.content_note": (
            "The two methods show moderate consistency, "
            "but similarity scores are not booking "
            "probabilities."
        ),
        "evaluation.no_ground_truth": (
            "Precision@K and Recall@K cannot be calculated "
            "because the dataset has no recommendation "
            "ground truth."
        ),
        "evaluation.collaborative": (
            "Collaborative Filtering"
        ),
        "evaluation.ridge_selected": (
            "One-Hot + Ridge is used in the GUI because "
            "it achieved the lowest RMSE and MAE "
            "on the test set."
        ),
        "evaluation.als_note": (
            "PySpark ALS is still implemented and evaluated "
            "in the notebook to satisfy the Big Data in "
            "Machine Learning requirement, but it does not "
            "run directly inside the Streamlit app."
        ),
        "evaluation.limitations": "Limitations",

        "limitation.single_interaction": (
            "Each Reviewer ID has only one interaction."
        ),
        "limitation.group_level": (
            "Collaborative recommendations operate at "
            "customer-group level rather than individual level."
        ),
        "limitation.hotel_id": (
            "Some Hotel IDs do not match between the two tables."
        ),
        "limitation.no_ground_truth": (
            "There is no recommendation ground truth "
            "for calculating Precision@K or Recall@K."
        ),

        # ----------------------------------------------------
        # Hotel cards
        # ----------------------------------------------------
        "card.image_caption": "Illustrative image",
        "card.stars": "{value} stars",
        "card.review_count": "{count} reviews",
        "card.no_address": (
            "📍 Address not available."
        ),
        "card.match_score": "Match score",
        "card.similarity_score": "Similarity",
        "card.predicted_score": "Predicted score",
        "card.details": "Details & Reviews",
        "card.rating_details": "Detailed ratings",
        "card.description": "Hotel description",
        "card.no_description": (
            "No detailed hotel description is available."
        ),
        "card.customer_reviews": "Guest reviews",
        "card.dataset_reviews": (
            "Reviews available in dataset: {count}"
        ),
        "card.no_reviews": (
            "This hotel has no reviews in the dataset."
        ),
        "card.high_reviews": (
            "👍 Top 5 highest-scoring reviews"
        ),
        "card.low_reviews": (
            "👎 Top 5 lowest-scoring reviews"
        ),
        "card.no_matching_reviews": (
            "No matching reviews are available."
        ),

        # ----------------------------------------------------
        # Rating criteria
        # ----------------------------------------------------
        "criteria.location": "Location",
        "criteria.cleanliness": "Cleanliness",
        "criteria.service": "Service",
        "criteria.facilities": "Facilities",
        "criteria.value_for_money": "Value for money",
        "criteria.comfort_and_room_quality": (
            "Comfort & room quality"
        ),

        # ----------------------------------------------------
        # Hotel highlights
        # ----------------------------------------------------
        "highlight.apartment": "🏠 Apartment",
        "highlight.villa": "🏡 Villa",
        "highlight.homestay": "🛏️ Homestay",
        "highlight.house": "🏘️ Private house",
        "highlight.resort": "🌴 Resort",
        "highlight.near_beach": "🏖️ Near beach",
        "highlight.near_airport": "✈️ Near airport",
        "highlight.airport_transfer": (
            "✈️ Airport transfer"
        ),
        "highlight.pool": "🏊 Pool",
        "highlight.spa": "💆 Spa",
        "highlight.gym": "🏋️ Gym",
        "highlight.restaurant": "🍽️ Restaurant",
        "highlight.family": "👨‍👩‍👧 Family-friendly",
        "highlight.luxury": "✨ Luxury",

        "error.hotel_not_found": (
            "Hotel not found."
        ),

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------
        "error.query_required": (
            "Please describe the hotel you are looking for."
        ),

        "error.no_matching_keywords": (
            "No matching keywords were found "
            "in the hotel dataset. "
            "Try a different description."
        ),

        "error.no_candidates": (
            "No hotel satisfies all selected conditions. "
            "Try removing one of the conditions."
        ),

        "error.no_content_match": (
            "No hotel with matching content "
            "was found for this description."
        ),

        "error.no_gensim_recommendation": (
            "No Gensim recommendations are "
            "available for this hotel."
        ),

        "error.customer_group_not_found": (
            "Customer group not found."
        ),
    },
}


# ============================================================
# DataFrame display translations
# ============================================================

COLUMN_TRANSLATIONS = {
    "vi": {
        "score_level": "Mức đánh giá",
        "review_count": "Số review",
        "percentage": "Tỷ lệ (%)",
        "nationality": "Quốc tịch",
        "group_name": "Nhóm khách",
        "average_score": "Điểm trung bình",
        "review_year": "Năm",
        "keyword": "Từ khóa",
        "frequency": "Tần suất",
        "model": "Mô hình",
        "score": "Điểm",
        "title": "Tiêu đề",
        "body": "Nội dung",
        "review_date": "Ngày review",
    },

    "en": {
        "Thông tin": "Information",
        "Giá trị": "Value",
        "Tiêu chí": "Criterion",
        "Điểm khách sạn": "Hotel score",
        "TB toàn hệ thống": "System average",
        "TB cùng hạng sao": "Same-star average",
        "Chênh lệch với hệ thống": (
            "Difference vs. system"
        ),
        "Nhận xét": "Assessment",

        "score_level": "Score level",
        "review_count": "Reviews",
        "percentage": "Percentage (%)",
        "nationality": "Nationality",
        "group_name": "Travel group",
        "average_score": "Average score",
        "review_year": "Year",
        "keyword": "Keyword",
        "frequency": "Frequency",
        "model": "Model",
        "score": "Score",
        "title": "Title",
        "body": "Review",
        "review_date": "Review date",
    },
}


VALUE_TRANSLATIONS_EN = {
    "Tên khách sạn": "Hotel name",
    "Hotel ID": "Hotel ID",
    "Hạng sao": "Star rating",
    "Địa chỉ": "Address",
    "Total Score": "Total score",
    "Điểm review trung bình": "Average review score",
    "Số review thực tế": "Actual review count",
    "comments_count": "Dataset comments_count",
    "Review đầu tiên": "First review",
    "Review gần nhất": "Latest review",

    "Không có dữ liệu": "No data",
    "Điểm mạnh": "Strength",
    "Cần cải thiện": "Needs improvement",
    "Gần trung bình": "Near average",

    "Vị trí": "Location",
    "Sạch sẽ": "Cleanliness",
    "Dịch vụ": "Service",
    "Tiện nghi": "Facilities",
    "Đáng giá tiền": "Value for money",
    "Thoải mái & chất lượng phòng": (
        "Comfort & room quality"
    ),
}


# ============================================================
# Helpers
# ============================================================

def t(
    key: str,
    language: str,
    **kwargs,
) -> str:
    """Return translated UI text."""

    language = (
        language
        if language in TRANSLATIONS
        else "vi"
    )

    text = (
        TRANSLATIONS[
            language
        ].get(
            key,
            TRANSLATIONS[
                "vi"
            ].get(
                key,
                key,
            ),
        )
    )

    if kwargs:
        return text.format(
            **kwargs
        )

    return text


def localize_value(
    value,
    language: str,
):
    """Translate selected display values only."""

    if language != "en":
        return value

    return VALUE_TRANSLATIONS_EN.get(
        value,
        value,
    )


def localize_dataframe(
    dataframe: pd.DataFrame,
    language: str,
    translate_values: bool = False,
) -> pd.DataFrame:
    """Return a localized copy for display only."""

    result = dataframe.copy()

    result = result.rename(
        columns=(
            COLUMN_TRANSLATIONS.get(
                language,
                {},
            )
        )
    )

    if (
        language == "en"
        and translate_values
    ):
        result = result.replace(
            VALUE_TRANSLATIONS_EN
        )

    return result


def validate_translations():
    """Ensure Vietnamese and English contain the same keys."""

    vi_keys = set(
        TRANSLATIONS["vi"]
    )

    en_keys = set(
        TRANSLATIONS["en"]
    )

    missing_in_vi = (
        en_keys - vi_keys
    )

    missing_in_en = (
        vi_keys - en_keys
    )

    if (
        missing_in_vi
        or missing_in_en
    ):
        raise ValueError(
            "Translation keys do not match. "
            f"Missing in vi: {sorted(missing_in_vi)}; "
            f"Missing in en: {sorted(missing_in_en)}"
        )