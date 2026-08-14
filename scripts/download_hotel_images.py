import argparse
import csv
import json
import os

from pathlib import Path
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.parse import (
    urlencode,
)
from urllib.request import (
    Request,
    urlopen,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

ASSET_DIR = (
    PROJECT_ROOT
    / "assets"
)

IMAGE_ROOT = (
    ASSET_DIR
    / "hotel_images"
)

HERO_DIR = (
    ASSET_DIR
    / "hero"
)

SOURCE_CSV = (
    ASSET_DIR
    / "image_sources.csv"
)


# ============================================================
# Pexels configuration
# ============================================================

PEXELS_SEARCH_URL = (
    "https://api.pexels.com/v1/search"
)

IMAGES_PER_CATEGORY = 10

SEARCH_RESULTS = 30

CARD_SIZE_KEY = "medium"

HERO_SIZE_KEY = "landscape"

CARD_WARNING_KB = 200

HERO_WARNING_KB = 500


CATEGORY_QUERIES = {
    "resort": (
        "tropical beach resort pool"
    ),
    "hotel": (
        "modern hotel exterior"
    ),
    "apartment": (
        "modern serviced apartment interior"
    ),
    "villa": (
        "private tropical villa pool"
    ),
    "homestay": (
        "cozy guest house homestay"
    ),
    "house": (
        "modern vacation house"
    ),
}


HERO_QUERY = (
    "tropical beach resort ocean"
)


CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


# ============================================================
# API helpers
# ============================================================

def search_photos(
    api_key: str,
    query: str,
    per_page: int,
):
    """Search landscape photos on Pexels."""

    parameters = {
        "query": query,
        "orientation": "landscape",
        "per_page": per_page,
        "locale": "en-US",
    }

    url = (
        PEXELS_SEARCH_URL
        + "?"
        + urlencode(
            parameters
        )
    )

    request = Request(
        url,
        headers={
            "Authorization": api_key,
            "User-Agent": (
                "hotel-recommendation-"
                "customer-insights"
            ),
        },
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:
            data = json.load(
                response
            )

            remaining = (
                response.headers.get(
                    "X-Ratelimit-Remaining"
                )
            )

    except HTTPError as error:
        raise RuntimeError(
            "Pexels API request failed: "
            f"HTTP {error.code}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Không thể kết nối Pexels API."
        ) from error

    photos = data.get(
        "photos",
        [],
    )

    return (
        photos,
        remaining,
    )


def select_unique_photos(
    photos,
    used_photo_ids: set,
    count: int,
):
    """Select unique Pexels photos."""

    selected = []

    for photo in photos:
        photo_id = photo.get(
            "id"
        )

        if (
            photo_id is None
            or photo_id
            in used_photo_ids
        ):
            continue

        selected.append(
            photo
        )

        used_photo_ids.add(
            photo_id
        )

        if len(selected) >= count:
            break

    return selected


# ============================================================
# Download helpers
# ============================================================

def download_photo(
    photo: dict,
    output_base: Path,
    size_key: str,
) -> Path:
    """Download one photo from the Pexels CDN."""

    image_url = (
        photo[
            "src"
        ][
            size_key
        ]
    )

    request = Request(
        image_url,
        headers={
            "User-Agent": (
                "hotel-recommendation-"
                "customer-insights"
            ),
        },
    )

    try:
        with urlopen(
            request,
            timeout=60,
        ) as response:
            image_bytes = (
                response.read()
            )

            content_type = (
                response.headers
                .get_content_type()
            )

    except HTTPError as error:
        raise RuntimeError(
            "Không thể tải ảnh Pexels: "
            f"HTTP {error.code}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Không thể tải ảnh từ Pexels."
        ) from error

    extension = (
        CONTENT_TYPE_EXTENSIONS.get(
            content_type
        )
    )

    if extension is None:
        raise RuntimeError(
            "Định dạng ảnh không hỗ trợ: "
            f"{content_type}"
        )

    output_path = (
        output_base
        .with_suffix(
            extension
        )
    )

    output_path.write_bytes(
        image_bytes
    )

    return output_path


def remove_generated_files():
    """Remove only images generated by this script."""

    for category in (
        CATEGORY_QUERIES
    ):
        category_dir = (
            IMAGE_ROOT
            / category
        )

        if not category_dir.exists():
            continue

        for path in category_dir.glob(
            f"{category}_*.*"
        ):
            path.unlink()

    if HERO_DIR.exists():
        for path in HERO_DIR.glob(
            "hotel_hero.*"
        ):
            path.unlink()

    if SOURCE_CSV.exists():
        SOURCE_CSV.unlink()


def find_existing_files():
    """Find files that this script would overwrite."""

    existing = []

    for category in (
        CATEGORY_QUERIES
    ):
        category_dir = (
            IMAGE_ROOT
            / category
        )

        if category_dir.exists():
            existing.extend(
                category_dir.glob(
                    f"{category}_*.*"
                )
            )

    if HERO_DIR.exists():
        existing.extend(
            HERO_DIR.glob(
                "hotel_hero.*"
            )
        )

    return list(
        existing
    )


# ============================================================
# Source metadata
# ============================================================

def build_source_record(
    photo: dict,
    image_path: Path,
    category: str,
):
    """Build one attribution/source row."""

    relative_path = (
        image_path
        .relative_to(
            ASSET_DIR
        )
        .as_posix()
    )

    return {
        "image_path": (
            relative_path
        ),
        "category": category,
        "pexels_photo_id": (
            photo.get(
                "id",
                "",
            )
        ),
        "photographer": (
            photo.get(
                "photographer",
                "",
            )
        ),
        "photographer_url": (
            photo.get(
                "photographer_url",
                "",
            )
        ),
        "source_page": (
            photo.get(
                "url",
                "",
            )
        ),
        "alt": (
            photo.get(
                "alt",
                "",
            )
        ),
    }


def save_source_csv(
    records,
):
    """Save Pexels photo provenance."""

    fieldnames = [
        "image_path",
        "category",
        "pexels_photo_id",
        "photographer",
        "photographer_url",
        "source_page",
        "alt",
    ]

    with open(
        SOURCE_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ============================================================
# Main download workflow
# ============================================================

def prepare_directories():
    """Create asset directories."""

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    HERO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for category in (
        CATEGORY_QUERIES
    ):
        (
            IMAGE_ROOT
            / category
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def download_category(
    api_key: str,
    category: str,
    query: str,
    used_photo_ids: set,
    source_records: list,
):
    """Download images for one hotel category."""

    print(
        f"\n[{category}] "
        f"Search: {query}"
    )

    photos, remaining = (
        search_photos(
            api_key=api_key,
            query=query,
            per_page=SEARCH_RESULTS,
        )
    )

    selected = (
        select_unique_photos(
            photos=photos,
            used_photo_ids=(
                used_photo_ids
            ),
            count=(
                IMAGES_PER_CATEGORY
            ),
        )
    )

    if (
        len(selected)
        < IMAGES_PER_CATEGORY
    ):
        raise RuntimeError(
            f"Không đủ ảnh unique cho "
            f"category '{category}'."
        )

    category_dir = (
        IMAGE_ROOT
        / category
    )

    for index, photo in enumerate(
        selected,
        start=1,
    ):
        output_base = (
            category_dir
            / (
                f"{category}_"
                f"{index:02d}"
            )
        )

        image_path = (
            download_photo(
                photo=photo,
                output_base=(
                    output_base
                ),
                size_key=(
                    CARD_SIZE_KEY
                ),
            )
        )

        size_kb = (
            image_path.stat().st_size
            / 1024
        )

        print(
            f"  {image_path.name}"
            f"  {size_kb:.0f} KB"
        )

        if (
            size_kb
            > CARD_WARNING_KB
        ):
            print(
                "    WARNING: "
                "ảnh lớn hơn "
                f"{CARD_WARNING_KB} KB"
            )

        source_records.append(
            build_source_record(
                photo=photo,
                image_path=image_path,
                category=category,
            )
        )

    if remaining is not None:
        print(
            "  API requests remaining:",
            remaining,
        )


def download_hero(
    api_key: str,
    used_photo_ids: set,
    source_records: list,
):
    """Download one wide hero image."""

    print(
        "\n[hero] "
        f"Search: {HERO_QUERY}"
    )

    photos, remaining = (
        search_photos(
            api_key=api_key,
            query=HERO_QUERY,
            per_page=10,
        )
    )

    selected = (
        select_unique_photos(
            photos=photos,
            used_photo_ids=(
                used_photo_ids
            ),
            count=1,
        )
    )

    if not selected:
        raise RuntimeError(
            "Không tìm được hero image."
        )

    photo = selected[0]

    image_path = (
        download_photo(
            photo=photo,
            output_base=(
                HERO_DIR
                / "hotel_hero"
            ),
            size_key=(
                HERO_SIZE_KEY
            ),
        )
    )

    size_kb = (
        image_path.stat().st_size
        / 1024
    )

    print(
        f"  {image_path.name}"
        f"  {size_kb:.0f} KB"
    )

    if (
        size_kb
        > HERO_WARNING_KB
    ):
        print(
            "    WARNING: hero lớn hơn "
            f"{HERO_WARNING_KB} KB"
        )

    source_records.append(
        build_source_record(
            photo=photo,
            image_path=image_path,
            category="hero",
        )
    )

    if remaining is not None:
        print(
            "  API requests remaining:",
            remaining,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download representative hotel "
            "images from Pexels."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace images generated "
            "by a previous run."
        ),
    )

    args = parser.parse_args()

    api_key = (
        os.environ.get(
            "PEXELS_API_KEY",
            "",
        )
        .strip()
    )

    if not api_key:
        raise SystemExit(
            "Thiếu PEXELS_API_KEY. "
            "Hãy export API key trước."
        )

    prepare_directories()

    existing_files = (
        find_existing_files()
    )

    if (
        existing_files
        and not args.overwrite
    ):
        raise SystemExit(
            "Đã có generated images. "
            "Nếu thật sự muốn tạo lại, "
            "chạy với --overwrite."
        )

    if args.overwrite:
        remove_generated_files()

    used_photo_ids = set()

    source_records = []

    for (
        category,
        query,
    ) in CATEGORY_QUERIES.items():

        download_category(
            api_key=api_key,
            category=category,
            query=query,
            used_photo_ids=(
                used_photo_ids
            ),
            source_records=(
                source_records
            ),
        )

    download_hero(
        api_key=api_key,
        used_photo_ids=(
            used_photo_ids
        ),
        source_records=(
            source_records
        ),
    )

    save_source_csv(
        source_records
    )

    print(
        "\n================================"
    )

    print(
        "Download completed."
    )

    print(
        "Hotel images:",
        len(source_records) - 1,
    )

    print(
        "Hero images: 1"
    )

    print(
        "Source metadata:",
        SOURCE_CSV,
    )


if __name__ == "__main__":
    main()