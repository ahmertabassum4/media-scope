"""Build the provenance feature dict the metadata factuality model was trained on.

Reuses the exact parsing/whois/tls logic from the training-time enricher
(``enrich_provenance.py``), vendored into this app package for the container.
"""
from app.enrich_provenance import (
    SUSPECT_TLDS,
    _tld,
    parse_html,
    reg_domain,
    tls_info,
    whois_age,
)


def build_provenance(html, final_url, host, timeout=15):
    """Assemble a provenance dict with the same nested keys the model was trained on."""
    parsed = parse_html(html, host)
    ext = _tld(host)
    prov = {
        "fetch_ok": True,
        "tld_suspect": ext.suffix.split(".")[-1] in SUSPECT_TLDS,
        "registered_domain": reg_domain(host),
        "transparency": parsed["transparency"],
        "composition": parsed["composition"],
        "content": parsed["content"],
        "archive": {"html_bytes": len(html.encode("utf-8"))},
    }
    prov.update(whois_age(host))      # registration_date, domain_age_days, registrar
    prov["tls"] = tls_info(host, timeout)
    return prov
