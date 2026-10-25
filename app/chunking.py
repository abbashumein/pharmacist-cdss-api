"""Split one openFDA drug label into small, searchable chunks.

Why: the old ingest glued every section of a label into ONE long text and cut
each section (Interactions at 1,200 characters). Two problems followed:
  1. The cut dropped facts. Stored warfarin chunks contained 0 mentions of
     aspirin, although the real label lists it.
  2. The embedding model (MiniLM) reads only about 256 tokens, so only the
     start of that long text (drug name + first part of Interactions) was
     searchable. Dosage and Warnings sections could not be found at all.

Now: one chunk per ~900 characters of each section, with overlap, and every
chunk starts with its own header "Drug: X | Section: Y |" so the embedding
describes exactly what the chunk is about.
"""
import re
from typing import Dict, List, Optional

MAX_CHUNK_CHARS = 900
OVERLAP_CHARS = 150
MAX_SECTION_CHARS = 12000   # safety cap per section (about 14 chunks)

# (section label, openFDA fields). Newer labels use warnings_and_cautions and
# boxed_warning; the old ingest read only "warnings" and missed them.
SECTIONS = [
    ("Interactions", ["drug_interactions"]),
    ("Contraindications", ["contraindications"]),
    ("Side Effects", ["adverse_reactions"]),
    ("Boxed Warning", ["boxed_warning"]),
    ("Warnings", ["warnings_and_cautions", "warnings"]),
    ("Dosage", ["dosage_and_administration"]),
    ("Indications", ["indications_and_usage"]),
    ("Pregnancy", ["pregnancy_or_breast_feeding", "pregnancy"]),
    ("Do Not Use", ["do_not_use"]),
    ("Stop Use", ["stop_use"]),
    ("Purpose", ["purpose"]),
]


def _text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(v) for v in value if v)
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _first(value) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def get_drug_name(record: dict) -> Dict[str, str]:
    """Return {"generic": ..., "brand": ...}; generic is "" if the label has no name."""
    openfda = record.get("openfda", {}) or {}
    generic = _first(openfda.get("generic_name"))
    brand = _first(openfda.get("brand_name"))
    active = _first(record.get("active_ingredient"))
    return {"generic": generic or brand or active, "brand": brand}


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """Windows of at most max_chars, cut at a sentence or space, with overlap."""
    text = _text(text)[:MAX_SECTION_CHARS]
    if len(text) <= max_chars:
        return [text] if text else []
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("; "))
            if cut < max_chars // 2:
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return [p for p in parts if p]


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def build_chunks(record: dict, record_index: int) -> List[dict]:
    """All chunks for one label: [{"id", "text", "metadata"}, ...]."""
    names = get_drug_name(record)
    generic, brand = names["generic"], names["brand"]
    if not generic:
        return []
    shown = f"{generic} ({brand})" if brand and brand.lower() != generic.lower() else generic

    chunks = []
    for label, fields in SECTIONS:
        body = " ".join(t for t in (_text(record.get(f)) for f in fields) if t)
        for part, piece in enumerate(split_text(body)):
            chunks.append({
                "id": f"r{record_index}-{_slug(label)}-{part}",
                "text": f"Drug: {shown} | Section: {label} | {piece}",
                "metadata": {"drug": generic.lower(), "section": label.lower(), "part": part},
            })
    return chunks
