import csv
import hashlib
import re
import unicodedata

from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "assets"
    / "hotel_images"
)

EXACT_IMAGE_MAP_PATH = (
    IMAGE_ROOT
    / "hotel_image_map.csv"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# Property categories
# ============================================================

CATEGORY_RULES = [
    (
        "apartment",
        (
            "can ho",
            "apartment",
            "condo",
            "condotel",
        ),
    ),
    (
        "villa",
        (
            "biet thu",
            "villa",
        ),
    ),
    (
        "homestay",
        (
            "homestay",
        ),
    ),
    (
        "house",
        (
            "nha rieng",
            "house",
        ),
    ),
    (
        "resort",
        (
            "resort",
            "khu nghi duong",
        ),
    ),
]


# ============================================================
# Text helpers
# ============================================================

def _normalize_name(
    value,
) -> str:
    """Normalize hotel name for category matching."""

    if value is None:
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


def infer_image_category(
    hotel_name,
) -> str:
    """Infer representative image category from hotel name."""

    name_text = (
        _normalize_name(
            hotel_name
        )
    )

    for category, terms in CATEGORY_RULES:
        if any(
            term in name_text
            for term in terms
        ):
            return category

    return "hotel"


# ============================================================
# Image files
# ============================================================

@lru_cache(maxsize=None)
def _get_category_images(
    category: str,
) -> tuple[Path, ...]:
    """Return local images for one category."""

    category_dir = (
        IMAGE_ROOT
        / category
    )

    if not category_dir.exists():
        return ()

    images = [
        path
        for path in category_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    return tuple(
        sorted(
            images
        )
    )


# ============================================================
# Optional exact-image override
# ============================================================

@lru_cache(maxsize=1)
def _load_exact_image_map() -> dict:
    """Load optional exact hotel image overrides."""

    if not EXACT_IMAGE_MAP_PATH.exists():
        return {}

    image_map = {}

    with open(
        EXACT_IMAGE_MAP_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            hotel_id = (
                row.get(
                    "hotel_id",
                    "",
                )
                .strip()
            )

            image_path = (
                row.get(
                    "image_path",
                    "",
                )
                .strip()
            )

            is_exact = (
                row.get(
                    "is_exact",
                    "",
                )
                .strip()
                .lower()
            )

            if (
                not hotel_id
                or not image_path
                or is_exact
                not in {
                    "true",
                    "1",
                    "yes",
                }
            ):
                continue

            full_path = (
                IMAGE_ROOT
                / image_path
            )

            if full_path.exists():
                image_map[
                    hotel_id
                ] = full_path

    return image_map


# ============================================================
# Deterministic mapping
# ============================================================

def _stable_image_index(
    hotel_id,
    image_count: int,
) -> int:
    """Map one hotel ID to a stable image index."""

    hotel_key = str(
        hotel_id
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        hotel_key
    ).hexdigest()

    number = int(
        digest[:12],
        16,
    )

    return (
        number
        % image_count
    )


def get_hotel_image(
    hotel_id,
    hotel_name,
):
    """Return image path and whether it is an exact hotel image."""

    exact_map = (
        _load_exact_image_map()
    )

    exact_image = (
        exact_map.get(
            str(hotel_id)
        )
    )

    if exact_image is not None:
        return (
            exact_image,
            True,
        )

    category = (
        infer_image_category(
            hotel_name
        )
    )

    images = (
        _get_category_images(
            category
        )
    )

    # Fallback về generic hotel.
    if (
        not images
        and category != "hotel"
    ):
        images = (
            _get_category_images(
                "hotel"
            )
        )

    if not images:
        return (
            None,
            False,
        )

    image_index = (
        _stable_image_index(
            hotel_id=hotel_id,
            image_count=len(
                images
            ),
        )
    )

    return (
        images[
            image_index
        ],
        False,
    )