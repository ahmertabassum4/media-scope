import argparse
import json
import re
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata" / "raw_site_metadata.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "features" / "site_metadata_features.jsonl"

TRACKER_HOSTS = (
    "google-analytics.com",
    "googletagmanager.com",
    "facebook.net",
    "facebook.com/tr",
    "scorecardresearch.com",
    "chartbeat.com",
    "parsely.com",
    "hotjar.com",
    "newrelic.com",
    "quantserve.com",
    "segment.com",
)
AD_HOSTS = (
    "doubleclick.net",
    "googlesyndication.com",
    "adservice.google",
    "adnxs.com",
    "taboola.com",
    "outbrain.com",
    "amazon-adsystem.com",
    "pubmatic.com",
    "openx.net",
    "rubiconproject.com",
)
SOCIAL_HOSTS = (
    "facebook.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "linkedin.com",
    "telegram.org",
    "bitchute.com",
    "rumble.com",
)


def present(value):
    if value is None:
        return "UNCLEAR"
    return "PRESENT" if bool(value) else "ABSENT"


def number(value, default=0):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return float(value)
    except Exception:
        return default


def rate(part, total):
    total = number(total)
    return round(number(part) / total, 5) if total else 0


def host_matches(host, patterns):
    host = str(host or "").lower()
    return any(p in host for p in patterns)


def flatten_hosts(hosts):
    return [str(h).lower() for h in hosts or [] if h]


def parsed_scheme(url):
    return urllib.parse.urlparse(str(url or "")).scheme.lower()


def clean_label(value):
    return str(value or "").upper().strip()


def build_features(row):
    html = row.get("html") if isinstance(row.get("html"), dict) else {}
    ssl_info = row.get("ssl") if isinstance(row.get("ssl"), dict) else {}
    requested_domain = row.get("requested_domain") if isinstance(row.get("requested_domain"), dict) else {}
    final_domain = row.get("final_domain") if isinstance(row.get("final_domain"), dict) else {}
    special = html.get("special_links") if isinstance(html.get("special_links"), dict) else {}
    script_hosts = flatten_hosts(html.get("script_hosts"))
    stylesheet_hosts = flatten_hosts(html.get("stylesheet_hosts"))
    image_hosts = flatten_hosts(html.get("image_hosts"))
    link_hosts = flatten_hosts(html.get("link_hosts"))
    meta_names = [str(m).lower() for m in html.get("meta_names", [])]
    schema_types = [str(s).lower() for s in html.get("schema_types", [])]
    title_length = number(html.get("title_length"))
    description_length = number(html.get("meta_description_length"))
    status_code = int(number(row.get("status_code")))
    redirect_count = int(number(row.get("redirect_count")))
    links_total = number(html.get("links_total"))
    internal_links = number(html.get("internal_links"))
    external_links = number(html.get("external_links"))
    script_count = number(html.get("script_count"))
    stylesheet_count = number(html.get("stylesheet_count"))
    image_count = number(html.get("image_count"))
    tracker_count = sum(1 for h in script_hosts if host_matches(h, TRACKER_HOSTS))
    ad_count = sum(1 for h in script_hosts if host_matches(h, AD_HOSTS))
    social_count = sum(1 for h in set(script_hosts + image_hosts + link_hosts) if host_matches(h, SOCIAL_HOSTS))
    final_url = row.get("final_url", "")
    requested_url = row.get("requested_url", "")

    categorical = {
        "m001_requested_https": present(parsed_scheme(requested_url) == "https"),
        "m002_final_https": present(parsed_scheme(final_url) == "https"),
        "m003_status_200": present(status_code == 200),
        "m004_status_error": present(status_code >= 400 if status_code else None),
        "m005_redirected": present(redirect_count > 0 or requested_url != final_url),
        "m006_ssl_valid": present(ssl_info.get("valid") if ssl_info.get("checked") else None),
        "m007_ssl_expiring_soon": present(number(ssl_info.get("days_left"), 9999) < 30 if ssl_info.get("checked") else None),
        "m008_has_title": present(title_length > 0),
        "m009_has_meta_description": present(description_length > 0),
        "m010_has_canonical": present(html.get("has_canonical")),
        "m011_has_og_tags": present(number(html.get("og_tag_count")) > 0),
        "m012_has_twitter_card": present(number(html.get("twitter_tag_count")) > 0),
        "m013_has_schema_org": present(html.get("has_schema_org")),
        "m014_has_news_schema": present(html.get("has_news_schema")),
        "m015_has_rss_or_atom": present(html.get("has_rss_or_atom")),
        "m016_has_about_link": present(special.get("about")),
        "m017_has_contact_link": present(special.get("contact")),
        "m018_has_privacy_link": present(special.get("privacy")),
        "m019_has_terms_link": present(special.get("terms")),
        "m020_has_advertise_link": present(special.get("advertise")),
        "m021_has_masthead_or_staff_link": present(special.get("masthead")),
        "m022_has_corrections_or_policy_link": present(special.get("corrections")),
        "m023_has_subscribe_link": present(special.get("subscribe")),
        "m024_has_donate_link": present(special.get("donate")),
        "m025_has_login_link": present(special.get("login")),
        "m026_wordpress_detected": present(any("wp-" in h or "wordpress" in h for h in script_hosts + stylesheet_hosts)),
        "m027_amp_detected": present(html.get("amp_detected")),
        "m028_google_analytics_detected": present(any("google-analytics.com" in h for h in script_hosts)),
        "m029_google_tag_manager_detected": present(any("googletagmanager.com" in h for h in script_hosts)),
        "m030_ad_network_detected": present(ad_count > 0),
        "m031_tracker_detected": present(tracker_count > 0),
        "m032_paywall_or_meter_hint": present(any(re.search(r"(piano|tinypass|meter|paywall|subscribe)", h) for h in script_hosts)),
        "m033_has_search": present(html.get("has_search")),
        "m034_author_schema_hint": present(any("person" in s for s in schema_types)),
        "m035_publisher_schema_hint": present(any("organization" in s or "newspaper" in s for s in schema_types)),
    }

    numeric = {
        "status_code": status_code,
        "redirect_count": redirect_count,
        "elapsed_ms": number(row.get("elapsed_ms")),
        "bytes_read": number(row.get("bytes_read")),
        "requested_domain_length": number(requested_domain.get("domain_length")),
        "final_domain_length": number(final_domain.get("domain_length")),
        "subdomain_count": number(final_domain.get("subdomain_count")),
        "hyphen_count": number(final_domain.get("hyphen_count")),
        "digit_count": number(final_domain.get("digit_count")),
        "path_length": number(final_domain.get("path_length")),
        "query_length": number(final_domain.get("query_length")),
        "ssl_days_left": number(ssl_info.get("days_left"), -1),
        "title_length": title_length,
        "meta_description_length": description_length,
        "meta_tag_count": len(meta_names),
        "og_tag_count": number(html.get("og_tag_count")),
        "twitter_tag_count": number(html.get("twitter_tag_count")),
        "schema_type_count": len(schema_types),
        "links_total": links_total,
        "internal_links": internal_links,
        "external_links": external_links,
        "external_link_rate": rate(external_links, links_total),
        "script_count": script_count,
        "external_script_host_count": len(set(script_hosts)),
        "stylesheet_count": stylesheet_count,
        "external_stylesheet_host_count": len(set(stylesheet_hosts)),
        "image_count": image_count,
        "external_image_host_count": len(set(image_hosts)),
        "tracker_script_count": tracker_count,
        "ad_script_count": ad_count,
        "social_asset_host_count": social_count,
        "word_count": number(html.get("word_count")),
        "heading_count": number(html.get("heading_count")),
        "h1_count": number(html.get("h1_count")),
        "h2_count": number(html.get("h2_count")),
        "h3_count": number(html.get("h3_count")),
        "avg_heading_length": number(html.get("avg_heading_length")),
        "uppercase_heading_rate": number(html.get("uppercase_heading_rate")),
        "article_tag_count": number(html.get("article_tag_count")),
        "nav_tag_count": number(html.get("nav_tag_count")),
        "form_count": number(html.get("form_count")),
    }

    return categorical, numeric


def convert_row(row):
    features, derived = build_features(row)
    return {
        "filename": f"{row.get('media_name', '')}.metadata",
        "ground_truth": clean_label(row.get("ground_truth")),
        "bias": clean_label(row.get("bias")),
        "ok": bool(row.get("ok")),
        "source_url": row.get("requested_url", ""),
        "final_url": row.get("final_url", ""),
        "parsed": {
            "outlet_type": "",
            "features": features,
            "derived": derived,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert raw site metadata into classifier features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only-ok", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if args.only_ok and not row.get("ok"):
                skipped += 1
                continue
            out.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} feature rows -> {args.output}")
    if skipped:
        print(f"Skipped {skipped} rows")


if __name__ == "__main__":
    main()
