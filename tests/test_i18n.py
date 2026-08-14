import unittest

from src.i18n import (
    TRANSLATIONS,
    t,
    validate_translations,
)

from src.recommenders import (
    RecommendationError,
)

class I18nTests(
    unittest.TestCase
):

    def test_translation_keys_match(
        self,
    ):
        validate_translations()

    def test_vietnamese_translation(
        self,
    ):
        self.assertEqual(
            t(
                "nav.search",
                "vi",
            ),
            "🔎 Tìm khách sạn",
        )

    def test_english_translation(
        self,
    ):
        self.assertEqual(
            t(
                "nav.search",
                "en",
            ),
            "🔎 Find Hotels",
        )

    def test_format_parameters(
        self,
    ):
        self.assertEqual(
            t(
                "search.found",
                "en",
                count=10,
            ),
            "Found 10 matching hotels.",
        )

    def test_languages_have_same_keys(
        self,
    ):
        self.assertEqual(
            set(
                TRANSLATIONS[
                    "vi"
                ]
            ),
            set(
                TRANSLATIONS[
                    "en"
                ]
            ),
        )

    def test_recommendation_error_key(
        self,
    ):
        error = RecommendationError(
            "error.query_required"
        )

        self.assertEqual(
            error.message_key,
            "error.query_required",
        )

        self.assertEqual(
            t(
                error.message_key,
                "en",
            ),
            (
                "Please describe the hotel "
                "you are looking for."
            ),
        )


if __name__ == "__main__":
    unittest.main()