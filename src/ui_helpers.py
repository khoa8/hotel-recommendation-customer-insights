import re
import unicodedata

import pandas as pd


# ============================================================
# Highlight rules
# ============================================================

# Property type chỉ đọc từ tên hotel.
# Thứ tự quan trọng: apartment phải đứng trước resort.
PROPERTY_RULES = [
    (
        "highlight.apartment",
        (
            "can ho",
            "apartment",
            "condo",
            "condotel",
        ),
    ),
    (
        "highlight.villa",
        (
            "biet thu",
            "villa",
        ),
    ),
    (
        "highlight.homestay",
        (
            "homestay",
        ),
    ),
    (
        "highlight.house",
        (
            "nha rieng",
            "house",
        ),
    ),
    (
        "highlight.resort",
        (
            "resort",
            "khu nghi duong",
        ),
    ),
]


# Chỉ dùng các cụm từ cho thấy hotel thật sự gần biển.
BEACH_TERMS = (
    "gan bien",
    "gan bai bien",
    "sat bien",
    "ngay ben bai bien",
    "cach bai bien vai buoc",
    "cach bien vai buoc",
    "beachfront",
    "private beach",
    "near beach",
    "near the beach",
)


# Không coi mọi từ "airport" là gần sân bay.
NEAR_AIRPORT_TERMS = (
    "gan san bay",
    "near airport",
    "near the airport",
)


AIRPORT_TRANSFER_TERMS = (
    "dua don san bay",
    "dua don tu san bay",
    "don tien san bay",
    "airport shuttle",
    "airport transfer",
)


AMENITY_RULES = [
    (
        "highlight.pool",
        (
            "khach san co ho boi",
            "khach san co be boi",
            "ho boi ngoai troi",
            "be boi ngoai troi",
            "ho boi trong nha",
            "be boi trong nha",
            "swimming pool",
        ),
    ),
    (
        "highlight.spa",
        (
            "spa cua khach san",
            "khach san co spa",
            "khach san co mot spa",
            "spa tai khach san",
            "on site spa",
        ),
    ),
    (
        "highlight.gym",
        (
            "phong tap the duc",
            "trung tam the duc",
            "phong gym",
            "fitness center",
            "gym cua khach san",
        ),
    ),
    (
        "highlight.restaurant",
        (
            "khach san co nha hang",
            "khach san co mot nha hang",
            "nha hang cua khach san",
            "nha hang trong khuon vien",
            "on site restaurant",
        ),
    ),
]


FAMILY_TERMS = (
    "than thien voi tre em",
    "gia dinh co tre",
    "phu hop cho gia dinh",
    "phu hop voi gia dinh",
    "family friendly",
)


LUXURY_NAME_TERMS = (
    "luxury",
    "cao cap",
)


# ============================================================
# Helper functions
# ============================================================

def _normalize_text(
    value,
) -> str:
    """Normalize Vietnamese text for simple keyword matching."""

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


def _contains_any(
    text: str,
    terms,
) -> bool:
    """Return True when text contains at least one term."""

    return any(
        term in text
        for term in terms
    )


def _get_property_highlight(
    hotel_name: str,
):
    """Infer a user-facing property tag from hotel name."""

    for label, terms in PROPERTY_RULES:
        if _contains_any(
            hotel_name,
            terms,
        ):
            return label

    return None


def extract_hotel_highlights(
    hotel_name,
    hotel_description,
    hotel_address,
    max_tags: int = 5,
) -> list[str]:
    """Extract conservative user-facing hotel highlights."""

    if max_tags <= 0:
        return []

    name_text = _normalize_text(
        hotel_name
    )

    description_text = _normalize_text(
        hotel_description
    )

    address_text = _normalize_text(
        hotel_address
    )

    full_text = " ".join(
        [
            name_text,
            description_text,
            address_text,
        ]
    )

    highlights = []

    # --------------------------------------------------------
    # Property type
    # --------------------------------------------------------

    property_highlight = (
        _get_property_highlight(
            name_text
        )
    )

    if property_highlight:
        highlights.append(
            property_highlight
        )

    # --------------------------------------------------------
    # Beach
    # --------------------------------------------------------

    if _contains_any(
        full_text,
        BEACH_TERMS,
    ):
        highlights.append(
            "highlight.near_beach"
        )

    # --------------------------------------------------------
    # Airport
    # --------------------------------------------------------

    if _contains_any(
        full_text,
        NEAR_AIRPORT_TERMS,
    ):
        highlights.append(
            "highlight.near_airport"
        )

    elif _contains_any(
        full_text,
        AIRPORT_TRANSFER_TERMS,
    ):
        highlights.append(
            "highlight.airport_transfer"
        )

    # --------------------------------------------------------
    # Amenities
    # --------------------------------------------------------

    for label, terms in AMENITY_RULES:
        if _contains_any(
            full_text,
            terms,
        ):
            highlights.append(
                label
            )

    # --------------------------------------------------------
    # Family-friendly
    # --------------------------------------------------------

    if _contains_any(
        full_text,
        FAMILY_TERMS,
    ):
        highlights.append(
            "highlight.family"
        )

    # --------------------------------------------------------
    # Luxury
    # --------------------------------------------------------

    # Chỉ dựa trên tên hotel để tránh suy diễn
    # từ các câu marketing chung trong description.
    if _contains_any(
        name_text,
        LUXURY_NAME_TERMS,
    ):
        highlights.append(
            "highlight.luxury"
        )

    return highlights[:max_tags]