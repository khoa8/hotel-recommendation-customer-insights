import base64
import mimetypes

from pathlib import Path

import streamlit as st

from src.i18n import (
    t,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

STYLE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "styles.css"
)

HERO_DIR = (
    PROJECT_ROOT
    / "assets"
    / "hero"
)

SUPPORTED_HERO_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


@st.cache_data
def _load_hero_image_uri():
    """Load local hero image as a data URI."""

    if not HERO_DIR.exists():
        return ""

    hero_images = [
        path
        for path in HERO_DIR.iterdir()
        if (
            path.is_file()
            and path.name.startswith(
                "hotel_hero"
            )
            and path.suffix.lower()
            in SUPPORTED_HERO_EXTENSIONS
        )
    ]

    if not hero_images:
        return ""

    hero_path = sorted(
        hero_images
    )[0]

    mime_type = (
        mimetypes.guess_type(
            hero_path.name
        )[0]
        or "image/jpeg"
    )

    encoded_image = (
        base64.b64encode(
            hero_path.read_bytes()
        )
        .decode(
            "ascii"
        )
    )

    return (
        f"data:{mime_type};base64,"
        f"{encoded_image}"
    )


def load_app_styles():
    """Load custom CSS for the Streamlit app."""

    css = STYLE_PATH.read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def display_global_banner(language:str,):
    """Display the global hotel discovery hero."""

    image_uri = (
        _load_hero_image_uri()
    )

    inline_style = ""

    if image_uri:
        inline_style = (
            "background-image:"
            "linear-gradient("
            "90deg,"
            "rgba(9, 38, 71, 0.72) 0%,"
            "rgba(9, 38, 71, 0.40) 52%,"
            "rgba(9, 38, 71, 0.16) 100%"
            "),"
            f"url('{image_uri}');"
        )

    banner_html = (
        '<div class="hotel-hero" '
        f'style="{inline_style}">'
        '  <div class="hotel-hero-title">'
        f'    {t("hero.title", language)}'
        '  </div>'
        '</div>'
    )

    st.markdown(
        banner_html,
        unsafe_allow_html=True,
    )