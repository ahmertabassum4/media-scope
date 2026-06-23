import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse the exact parsing/whois/tls logic from the training-time enricher.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from enrich_provenance import parse_html, whois_age, tls_info, reg_domain, _tld, SUSPECT_TLDS


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
