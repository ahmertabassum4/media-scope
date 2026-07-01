import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "s004_sensational_verbs", "s008_neutral_headlines", "s010_scare_quotes",
    "s023_no_owner_identified", "s060_templated_imagery", "s066_cluttered_blog_look",
    "s067_alarmist_colors", "s072_loaded_category_tags", "s076_conspiracy_tropes",
    "s078_absolutist_framing", "s085_standard_sections", "s097_suspicious_url",
    "nav_item_count", "footer_link_count", "story_block_count", "side_margin_frac",
    "content_column_count", "hero_image_area_frac", "largest_face_area_frac",
    "header_band_height_frac", "inter_story_whitespace_frac", "cta_button_count",
    "newsletter_box_count", "social_icon_count", "search_box_present",
    "weather_widget_present", "account_button_present", "category_badge_count",
    "accent_hue", "accent_color_count", "background_brightness", "logo_centered",
    "headline_size_variance", "thumbnail_size_uniformity", "mean_headline_word_count",
    "page_text_density", "timestamp_count", "allcaps_token_frac", "number_token_frac",
    "page_scroll_length", "small_face_count", "city_or_county_in_name",
    "lx_ideology_identity_per100", "lx_attribution_per100", "lx_timestamp_per100",
    "lx_loaded_outgroup_per100", "lx_health_scare_per100", "lx_question_style_per100",
    "lx_culture_war_per100", "lx_market_money_per100", "lx_local_civic_per100",
)

# ordinal encodings for the categorical features
CATEGORICAL = {
    "content_column_count": {"1": 1, "2": 2, "3plus": 3},
    "page_scroll_length": {"short": 0, "medium": 1, "long": 2, "very_long": 3},
    "background_brightness": {"dark": 0, "gray": 1, "offwhite": 2, "white": 3},
    "page_text_density": {"sparse": 0, "medium": 1, "dense": 2},
    "accent_hue": {"none": 0, "neutral": 1, "blue": 2, "green": 3, "purple": 4, "orange": 5, "red": 6},
}


def _num(key, value):
    if value is None:
        return np.nan
    if key in CATEGORICAL:
        return float(CATEGORICAL[key].get(str(value).strip().lower(), np.nan))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def feature_frame(rows):
    records = [{k: _num(k, r["features"].get(k)) for k in FEATURE_NAMES} for r in rows]
    return pd.DataFrame(records, columns=list(FEATURE_NAMES))


def labels(rows):
    return pd.Series([r["label"] for r in rows], dtype=str)


def texts(rows):
    return pd.Series([r["text"] for r in rows], dtype=str)
