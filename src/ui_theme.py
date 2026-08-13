from pathlib import Path

import streamlit as st


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


def load_app_styles():
    """Load custom CSS for the Streamlit app."""

    css = STYLE_PATH.read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def display_global_banner():
    """Display a compact hotel-search banner on every page."""

    banner_html = (
        '<div class="hotel-hero">'
        '  <div class="hotel-hero-content">'
        '    <div class="hotel-hero-main">'
        '      <span class="hotel-hero-icon">🏨</span>'
        '      <div>'
        '        <div class="hotel-hero-title">'
        '          Tìm nơi ở phù hợp cho chuyến đi của bạn'
        '        </div>'
        '        <div class="hotel-hero-subtitle">'
        '          Tìm theo nhu cầu · Khách sạn tương tự · '
        '          Gợi ý theo nhóm khách · Đánh giá thực tế'
        '        </div>'
        '      </div>'
        '    </div>'
        '  </div>'
        '</div>'
    )

    st.markdown(
        banner_html,
        unsafe_allow_html=True,
    )