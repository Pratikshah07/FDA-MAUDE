"""
Sheet Analyser — accepts an arbitrary user-uploaded MAUDE-like CSV/XLSX,
uses Groq to map its columns to canonical names (IMDRF Code, Manufacturer,
Date Received), then rewrites the file in canonical form so the existing
IMDRF Insights pipeline can analyse it unchanged.
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

from backend.groq_client import GroqClient


CANONICAL_IMDRF = "IMDRF Code"
CANONICAL_MFR = "Manufacturer"
CANONICAL_DATE = "Date Received"


def _read_headers_and_sample(path: str, n_rows: int = 5) -> Tuple[List[str], List[Dict[str, str]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str, keep_default_na=False, nrows=n_rows + 1)
    elif ext == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         on_bad_lines="skip", nrows=n_rows + 1)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    headers = [str(c) for c in df.columns]
    sample = df.head(n_rows).fillna("").astype(str).to_dict(orient="records")
    return headers, sample


def _heuristic_map(headers: List[str], sample: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """Cheap deterministic pass before calling the LLM."""
    norm = {h: h.strip().lower() for h in headers}
    result = {"imdrf_code": None, "manufacturer": None, "date_received": None, "event_date": None}

    for h, n in norm.items():
        if result["imdrf_code"] is None and "imdrf" in n:
            result["imdrf_code"] = h
        if result["manufacturer"] is None and ("manufacturer" in n or n in {"mfr", "company"}):
            result["manufacturer"] = h
        if result["date_received"] is None and "received" in n and "date" in n:
            result["date_received"] = h
        if result["event_date"] is None and "event" in n and "date" in n:
            result["event_date"] = h

    if result["imdrf_code"] is None:
        code_pat = re.compile(r"^[A-Za-z]\d{2}")
        for h in headers:
            vals = [str(row.get(h, "")) for row in sample if row.get(h)]
            hits = sum(1 for v in vals if code_pat.match(re.sub(r"[^A-Za-z0-9]", "", v)))
            if vals and hits / len(vals) > 0.4:
                result["imdrf_code"] = h
                break
    return result


def _groq_map(headers: List[str], sample: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    try:
        client = GroqClient()
    except Exception:
        return {"imdrf_code": None, "manufacturer": None, "date_received": None, "event_date": None}
    if not client.available:
        return {"imdrf_code": None, "manufacturer": None, "date_received": None, "event_date": None}

    headers_str = ", ".join(f'"{h}"' for h in headers)
    sample_str = json.dumps(sample[:3], ensure_ascii=False)[:1200]

    prompt = f"""You are mapping columns of a MAUDE-like medical-device adverse-event sheet.

Available headers (use EXACT strings, or null):
{headers_str}

First 3 sample rows (JSON):
{sample_str}

Identify which header best matches each canonical field:
- imdrf_code: column holding IMDRF codes (alphanumeric like "A05", "E2401", "A050101", may be pipe-separated)
- manufacturer: company / manufacturer name
- date_received: date the report was received (preferred date column)
- event_date: date the event occurred (secondary, optional)

Rules:
- Return ONLY a valid JSON object with exactly these keys.
- Use the EXACT header string from the list above, or null if absent.
- Do not invent or modify column names.

Return JSON only:
{{"imdrf_code": "...", "manufacturer": "...", "date_received": "...", "event_date": "..."}}"""

    try:
        resp = client.client.chat.completions.create(
            model=client.model,
            messages=[
                {"role": "system", "content": "You are a data schema expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        out = {}
        for key in ("imdrf_code", "manufacturer", "date_received", "event_date"):
            v = parsed.get(key)
            out[key] = v if (v in headers) else None
        return out
    except Exception as e:
        print(f"Sheet Analyser: Groq mapping failed: {e}")
        return {"imdrf_code": None, "manufacturer": None, "date_received": None, "event_date": None}


def ai_map_columns(file_path: str) -> Dict:
    """Inspect a sheet and return suggested column mapping plus all headers."""
    headers, sample = _read_headers_and_sample(file_path)
    if not headers:
        raise ValueError("Uploaded file has no columns.")

    detected = _heuristic_map(headers, sample)
    missing = [k for k, v in detected.items() if v is None]
    if missing:
        ai_result = _groq_map(headers, sample)
        for k in missing:
            if ai_result.get(k):
                detected[k] = ai_result[k]

    return {
        "headers": headers,
        "sample": sample,
        "mapping": detected,
    }


def rewrite_canonical(file_path: str, mapping: Dict[str, Optional[str]], output_path: str) -> Dict:
    """
    Rename the user's columns to canonical MAUDE names so the IMDRF insights
    pipeline can consume the file. Writes the canonicalised result as CSV.
    """
    imdrf_col = mapping.get("imdrf_code")
    if not imdrf_col:
        raise ValueError("IMDRF Code column is required — please pick the column that holds IMDRF codes.")

    date_col = mapping.get("date_received") or mapping.get("event_date")
    if not date_col:
        raise ValueError("A date column is required — please pick a 'Date Received' or 'Event Date' column.")

    mfr_col = mapping.get("manufacturer")  # optional — pipeline tolerates absence

    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    elif ext == ".csv":
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace(
            {"nan": "", "NaN": "", "None": "", "NaT": "", "<NA>": ""}
        )

    rename_map = {imdrf_col: CANONICAL_IMDRF, date_col: CANONICAL_DATE}
    if mfr_col and mfr_col in df.columns:
        rename_map[mfr_col] = CANONICAL_MFR
    df = df.rename(columns=rename_map)

    if CANONICAL_MFR not in df.columns:
        df[CANONICAL_MFR] = "All Data"

    df.to_csv(output_path, index=False, encoding="utf-8")

    return {
        "rows": len(df),
        "canonical_path": output_path,
        "applied_mapping": {
            "imdrf_code": imdrf_col,
            "manufacturer": mfr_col,
            "date_received": date_col,
        },
    }
