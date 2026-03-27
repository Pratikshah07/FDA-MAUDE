"""
IMDRF Prefix Insights Analysis Module

This module provides functionality to analyze IMDRF codes at the prefix level,
comparing manufacturers against universal and prefix-specific baselines.

Supports three analysis levels:
- Level-1: First 3 alphanumeric characters (e.g., "A01", "E24")
- Level-2: First 5 alphanumeric characters (e.g., "A0101", "E2401")
- Level-3: First 7 alphanumeric characters (e.g., "A010101", "E240101")

This is a read-only analysis module that does not modify the original data.
"""

import os
import re
from typing import Dict
from datetime import datetime
import pandas as pd
import numpy as np
from backend.parent_company_map import apply_parent_map

# Import the Annex validator for Level-2/3 validation
try:
    from backend.imdrf_annex_validator import get_validator, extract_imdrf_codes_by_level
except ImportError:
    from imdrf_annex_validator import get_validator, extract_imdrf_codes_by_level


# Level configuration
LEVEL_CONFIG = {
    1: {'length': 3, 'label': 'Level-1 (3 chars)'},
    2: {'length': 5, 'label': 'Level-2 (5 chars)'},
    3: {'length': 7, 'label': 'Level-3 (7 chars)'}
}

# IMDRF Level-1 prefixes that must be excluded from all code-count outputs.
# Codes at any level whose first 3 chars match an entry here will be dropped.
EXCLUDED_IMDRF_L1_PREFIXES: set = {'A24', 'A25'}


def parse_flexible_date(s):
    """
    Parse date string in multiple formats to pandas Timestamp.

    Supports:
    - DD-MM-YYYY (e.g., 01-04-2024)
    - YYYYMMDD (e.g., 20240401)
    - YYYY-MM-DD (e.g., 2024-04-01)
    - MM/DD/YYYY (e.g., 04/01/2024)
    - DD/MM/YYYY (e.g., 01/04/2024)

    Args:
        s: Date string in various formats

    Returns:
        pandas Timestamp or NaT if parsing fails
    """
    import pandas as pd

    if pd.isna(s):
        return pd.NaT

    s = str(s).strip()
    if not s or s.lower() in ('nan', 'nat', 'none', ''):
        return pd.NaT

    # Try multiple date formats
    formats = [
        '%d-%m-%Y',   # DD-MM-YYYY
        '%Y%m%d',     # YYYYMMDD
        '%Y-%m-%d',   # YYYY-MM-DD
        '%m/%d/%Y',   # MM/DD/YYYY
        '%d/%m/%Y',   # DD/MM/YYYY
        '%Y/%m/%d',   # YYYY/MM/DD
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue

    # Last resort: let pandas try to infer
    try:
        return pd.to_datetime(s, dayfirst=True)
    except Exception:
        return pd.NaT


def parse_ddmmyyyy_to_date(s):
    """
    Parse date string to pandas Timestamp using flexible parsing.
    (Wrapper for backward compatibility)

    Args:
        s: Date string in various formats

    Returns:
        pandas.Timestamp or pd.NaT if invalid
    """
    return parse_flexible_date(s)


def extract_imdrf_prefixes(imdrf_code_str, level=1):
    """
    Extract IMDRF prefixes from a code string at the specified level.

    Rules:
    - Split on "|" (pipe)
    - For each token, extract first N alphanumeric characters based on level
    - Uppercase the result
    - Skip if fewer than required alphanumeric characters

    Level-specific behavior:
    - Level-1: 3 characters (e.g., "A05")
    - Level-2: 5 characters (e.g., "A0501"), exact match only (does NOT include Level-3 codes)
    - Level-3: 7 characters (e.g., "A050101"), only exact matches if Annex validation enabled

    Args:
        imdrf_code_str: String containing IMDRF codes (may be pipe-separated)
        level: 1, 2, or 3 (default: 1 for backward compatibility)

    Returns:
        list of prefixes at the specified level
    """
    if pd.isna(imdrf_code_str):
        return []

    s = str(imdrf_code_str).strip()
    if not s or s.lower() in ["nan", "nat", "none", ""]:
        return []

    # Use the Annex validator extraction function for full validation
    return extract_imdrf_codes_by_level(s, level)


def extract_imdrf_prefixes_legacy(imdrf_code_str):
    """
    Legacy function for backward compatibility - extracts Level-1 (3 char) prefixes.

    Args:
        imdrf_code_str: String containing IMDRF codes (may be pipe-separated)

    Returns:
        list of prefixes (e.g., ["A05", "A07"])
    """
    return extract_imdrf_prefixes(imdrf_code_str, level=1)


def explode_imdrf_prefixes(df, imdrf_col, date_col, level=1):
    """
    Explode rows so each IMDRF prefix gets its own row.

    Args:
        df: DataFrame with IMDRF codes
        imdrf_col: Name of the IMDRF Code column
        date_col: Name of the date column to parse
        level: 1, 2, or 3 - determines prefix extraction length (default: 1)

    Returns:
        DataFrame with additional columns: 'imdrf_prefix', 'parsed_date'
        Only includes rows with valid IMDRF prefixes
    """
    df = df.copy()
    df['_prefixes'] = df[imdrf_col].apply(lambda x: extract_imdrf_prefixes(x, level=level))

    # Filter to rows with at least one prefix
    df_with_prefixes = df[df['_prefixes'].apply(len) > 0].copy()

    if df_with_prefixes.empty:
        return pd.DataFrame()

    # Explode so each prefix gets its own row
    df_exploded = df_with_prefixes.explode('_prefixes')
    df_exploded = df_exploded.rename(columns={'_prefixes': 'imdrf_prefix'})

    # Parse dates
    df_exploded['parsed_date'] = df_exploded[date_col].apply(parse_ddmmyyyy_to_date)

    return df_exploded


def aggregate_by_grain(df, date_col, grain='ME'):
    """
    Aggregate counts by date grain.

    Args:
        df: DataFrame with parsed dates
        date_col: Name of the parsed date column
        grain: pandas resample frequency — use 'ME' (monthly), 'QE' (quarterly).
               Legacy aliases 'M' and 'Q' are automatically mapped to their
               pandas 2.2+ equivalents.

    Returns:
        Series with date index and counts
    """
    # Map legacy frequency aliases to pandas 2.2+ equivalents
    _grain_map = {'M': 'ME', 'Q': 'QE'}
    grain = _grain_map.get(grain, grain)

    df = df[df[date_col].notna()].copy()

    if df.empty:
        return pd.Series(dtype=int)

    # Group by the specified grain
    counts = df.set_index(date_col).resample(grain).size()

    return counts


def find_date_column(df):
    """
    Find the appropriate date column with flexible matching.

    Searches for common date column names in priority order.
    """
    # Priority list of date column patterns (exact matches first, then contains)
    # "date received" is preferred over "event date" because it reflects the
    # actual intake date range selected by the user, whereas event date can
    # span arbitrary historical years outside the requested range.
    date_patterns_exact = [
        "date received", "date_received", "datereceived", "received_date",
        "report date", "report_date", "reportdate",
        "event date", "event_date", "eventdate",
        "date", "mdr_date"
    ]

    date_patterns_contains = [
        "received", "date", "event"
    ]

    # First pass: exact match (case-insensitive)
    for pattern in date_patterns_exact:
        for col in df.columns:
            col_lower = col.strip().lower().replace(" ", "_")
            if col_lower == pattern or col_lower.replace("_", "") == pattern.replace("_", ""):
                return col

    # Second pass: contains match
    for pattern in date_patterns_contains:
        for col in df.columns:
            if pattern in col.strip().lower():
                return col

    return None


def find_imdrf_column(df):
    """
    Find the IMDRF code column with flexible matching.

    Searches for IMDRF code columns only. Does NOT fall back to device_problem
    since that column contains descriptive text, not IMDRF codes.
    """
    import re

    # Priority list of IMDRF column patterns
    imdrf_patterns_exact = [
        "imdrf code", "imdrf_code", "imdrfcode",
        "imdrf", "imdrf codes", "imdrf_codes"
    ]

    imdrf_patterns_contains = ["imdrf"]

    # First pass: exact IMDRF match
    for pattern in imdrf_patterns_exact:
        for col in df.columns:
            col_lower = col.strip().lower().replace(" ", "_")
            if col_lower == pattern or col_lower.replace("_", "") == pattern.replace("_", ""):
                return col

    # Second pass: contains IMDRF
    for pattern in imdrf_patterns_contains:
        for col in df.columns:
            if pattern in col.strip().lower():
                return col

    # Third pass: look for columns that contain IMDRF-like codes (e.g., A05, E2401)
    # IMDRF codes start with a letter followed by digits
    imdrf_code_pattern = re.compile(r'^[A-Za-z]\d{2}')

    for col in df.columns:
        # Check if column values look like IMDRF codes
        sample_values = df[col].dropna().head(50)
        match_count = 0
        for val in sample_values:
            # Clean value and check pattern
            val_clean = re.sub(r'[^A-Za-z0-9]', '', str(val))
            if imdrf_code_pattern.match(val_clean) and len(val_clean) >= 3:
                match_count += 1

        # If more than 20% of samples match IMDRF pattern, use this column
        if len(sample_values) > 0 and match_count / len(sample_values) > 0.2:
            return col

    return None


def find_manufacturer_column(df):
    """
    Find the manufacturer column with flexible matching.

    If no manufacturer column exists, returns None (will use 'Unknown' as default).
    """
    # Priority list of manufacturer column patterns
    mfr_patterns_exact = [
        "manufacturer", "manufacturer name", "manufacturer_name",
        "manufacturername", "mfr", "mfr_name", "mfrname",
        "company", "company name", "company_name"
    ]

    mfr_patterns_contains = ["manufacturer", "mfr", "company"]

    # First pass: exact match
    for pattern in mfr_patterns_exact:
        for col in df.columns:
            col_lower = col.strip().lower().replace(" ", "_")
            if col_lower == pattern or col_lower.replace("_", "") == pattern.replace("_", ""):
                return col

    # Second pass: contains match
    for pattern in mfr_patterns_contains:
        for col in df.columns:
            if pattern in col.strip().lower():
                return col

    return None


def find_column(df, target):
    """Find a column by normalized name."""
    target_lower = target.strip().lower()
    for col in df.columns:
        if col.strip().lower() == target_lower:
            return col
    return None


def analyze_imdrf_insights(df, selected_prefix, selected_manufacturers, grain='M', level=1, date_from=None, date_to=None):
    """
    Perform IMDRF prefix insights analysis.

    Args:
        df: DataFrame with exploded IMDRF prefixes and parsed dates
        selected_prefix: IMDRF prefix to analyze (e.g., "A05" for Level-1, "A0501" for Level-2)
        selected_manufacturers: List of manufacturer names to compare
        grain: Date aggregation grain ('M' monthly, 'Q' quarterly)
        level: 1, 2, or 3 - IMDRF analysis level
        date_from: Optional ISO date string (YYYY-MM-DD) — used as the range start for the chart
        date_to: Optional ISO date string (YYYY-MM-DD) — used as the range end for the chart

    Returns:
        dict with analysis results including:
        - threshold: Percentage share of selected code vs all codes in dataset
        - manufacturer_series: Dict of time-series per manufacturer
        - date_range: Complete date range for plotting
        - statistics: Summary statistics per manufacturer
        - level: The analysis level used
    """
    mfr_col = None
    for col in df.columns:
        if col.strip().lower() in ["manufacturer", "manufacturer name"]:
            mfr_col = col
            break

    if mfr_col is None:
        raise ValueError("Manufacturer column not found")

    # Filter to rows with valid dates
    df_with_dates = df[df['parsed_date'].notna()].copy()

    if df_with_dates.empty:
        raise ValueError("No valid dates found in dataset")

    # Calculate threshold: (total count of selected code / total all codes) × 100
    total_all_codes = len(df_with_dates)
    df_prefix = df_with_dates[df_with_dates['imdrf_prefix'] == selected_prefix].copy()
    total_selected_code = len(df_prefix)
    threshold = round((total_selected_code / total_all_codes * 100) if total_all_codes > 0 else 0.0, 4)

    if df_prefix.empty:
        raise ValueError(f"No data found for prefix '{selected_prefix}'")

    # Filter to selected manufacturers
    df_selected = df_prefix[df_prefix[mfr_col].isin(selected_manufacturers)].copy()

    if df_selected.empty:
        raise ValueError(f"No data found for selected manufacturers with prefix '{selected_prefix}'")

    # Map legacy frequency aliases to pandas 2.2+ equivalents for resample/date_range
    _grain_map = {'M': 'ME', 'Q': 'QE'}
    pandas_grain = _grain_map.get(grain, grain)

    # Create time-series for each manufacturer
    manufacturer_series = {}

    for mfr in selected_manufacturers:
        df_mfr = df_selected[df_selected[mfr_col] == mfr]
        mfr_counts = aggregate_by_grain(df_mfr, 'parsed_date', pandas_grain)
        manufacturer_series[mfr] = mfr_counts

    # Build the full date range for the chart axes.
    # Priority: use the explicitly requested date_from / date_to (full year coverage),
    # then fall back to the span of all data for the selected code across ALL manufacturers.
    if date_from and date_to:
        try:
            date_range = pd.date_range(
                start=pd.to_datetime(date_from),
                end=pd.to_datetime(date_to),
                freq=pandas_grain
            )
        except Exception:
            date_range = pd.DatetimeIndex([])
    else:
        # Use all data for this code (not just selected mfrs) to determine the span
        all_code_counts = aggregate_by_grain(df_prefix, 'parsed_date', pandas_grain)
        if len(all_code_counts) > 0:
            date_range = pd.date_range(
                start=all_code_counts.index.min(),
                end=all_code_counts.index.max(),
                freq=pandas_grain
            )
        else:
            date_range = pd.DatetimeIndex([])

    # Reindex each manufacturer series to fill gaps with 0
    if len(date_range) > 0:
        for mfr in manufacturer_series:
            manufacturer_series[mfr] = manufacturer_series[mfr].reindex(date_range, fill_value=0)

    # Calculate summary statistics per manufacturer
    statistics = []
    for mfr, series in manufacturer_series.items():
        total_events = int(series.sum())
        mean_per_period = float(series.mean())
        max_per_period = int(series.max())
        periods_with_events = int((series > 0).sum())

        statistics.append({
            "manufacturer": mfr,
            "total_events": total_events,
            "mean_per_period": round(mean_per_period, 2),
            "max_per_period": max_per_period,
            "periods_with_events": periods_with_events
        })

    # Get level label for display
    level_label = LEVEL_CONFIG.get(level, {}).get('label', f'Level-{level}')

    return {
        "threshold": threshold,
        "total_selected_code": total_selected_code,
        "total_all_codes": total_all_codes,
        "manufacturer_series": manufacturer_series,
        "date_range": date_range,
        "statistics": statistics,
        "grain": grain,
        "selected_prefix": selected_prefix,
        "level": level,
        "level_label": level_label
    }


def prepare_data_for_insights(file_path, level=1, annex_file_path=None):
    """
    Read and prepare data for IMDRF insights analysis.

    Args:
        file_path: Path to cleaned CSV or Excel file
        level: 1, 2, or 3 - IMDRF analysis level (default: 1)

    Returns:
        dict with:
        - df_exploded: DataFrame with exploded IMDRF prefixes
        - all_prefixes: List of all available prefixes at the specified level
        - all_manufacturers: List of all manufacturers
        - date_col: Name of the date column used
        - level: The analysis level used
    """
    # Read file (CSV or Excel)
    import os
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    elif file_ext == '.csv':
        df = pd.read_csv(file_path, dtype=str, encoding="utf-8",
                         on_bad_lines="skip", keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please upload CSV, XLS, or XLSX file.")

    # Clean up string columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "NaN": "", "None": "", "NaT": "", "<NA>": ""})

    # Find required columns using flexible matching
    imdrf_col = find_imdrf_column(df)
    if imdrf_col is None:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(
            f"No IMDRF Code column found. This file does not appear to contain IMDRF codes. "
            f"IMDRF codes are alphanumeric (e.g., A05, E2401, A050101). "
            f"Please use a cleaned MAUDE file that has an 'IMDRF Code' column. "
            f"Available columns: {available_cols}"
        )

    # Find manufacturer column (optional - will use 'All Data' if not found)
    mfr_col = find_manufacturer_column(df)
    if mfr_col is None:
        # Create a synthetic manufacturer column with a default value
        df['_manufacturer'] = 'All Data'
        mfr_col = '_manufacturer'
        merge_log = []
    else:
        # Merge subsidiary/child company names into their parent (display-level only)
        df, merge_log = apply_parent_map(df, mfr_col)

    # Find date column
    date_col = find_date_column(df)
    if date_col is None:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(
            f"Missing date column. Expected columns like 'Event Date', 'Date Received', or 'date_received'. "
            f"Available columns: {available_cols}"
        )

    # Explode IMDRF prefixes at the specified level
    df_exploded = explode_imdrf_prefixes(df, imdrf_col, date_col, level=level)

    if df_exploded.empty:
        # Provide more helpful error message
        sample_values = df[imdrf_col].head(5).tolist()
        raise ValueError(
            f"No valid IMDRF codes found at Level-{level} in column '{imdrf_col}'. "
            f"Sample values: {sample_values}. "
            f"Level-{level} requires codes with at least {LEVEL_CONFIG[level]['length']} alphanumeric characters."
        )

    # Filter to rows with valid dates
    df_with_dates = df_exploded[df_exploded['parsed_date'].notna()].copy()

    # Exclude A24 and A25 codes (and their sub-codes at level 2/3) from all analysis
    df_with_dates = df_with_dates[
        ~df_with_dates['imdrf_prefix'].str[:3].str.upper().isin(EXCLUDED_IMDRF_L1_PREFIXES)
    ].copy()

    if df_with_dates.empty:
        sample_values = df[date_col].head(5).tolist()
        raise ValueError(
            f"No parsable dates found in '{date_col}' column. "
            f"Sample values: {sample_values}. "
            f"Supported formats: DD-MM-YYYY, YYYYMMDD, YYYY-MM-DD, MM/DD/YYYY"
        )

    # Get unique prefixes and manufacturers
    prefix_counts_series = df_with_dates['imdrf_prefix'].value_counts()
    # Convert to plain str keys to avoid float/str comparison errors in sorted()
    prefix_counts = {str(k): int(v) for k, v in prefix_counts_series.to_dict().items()
                     if str(k).strip() and str(k).strip().lower() not in ('nan', 'none', 'nat')}
    all_prefixes_set = set(prefix_counts.keys())

    # Also include patient problem E codes (from Annex E mapping) in the prefix list
    if annex_file_path:
        patient_col = None
        for col in df.columns:
            col_norm = str(col).strip().lower().replace('_', ' ')
            if col_norm in {"patient problem", "patient problems", "patient problem text"}:
                patient_col = col
                break
        if patient_col is None:
            for col in df.columns:
                col_norm = str(col).strip().lower().replace('_', ' ')
                if "patient" in col_norm and "problem" in col_norm:
                    patient_col = col
                    break
        if patient_col is not None:
            desc_to_code, _ = _build_e_desc_to_code_map(annex_file_path)
            prefix_len = LEVEL_CONFIG[level]['length']
            e_code_counts: Dict[str, int] = {}
            for raw in df[patient_col].dropna():
                raw = str(raw).strip()
                if not raw or raw.lower() in ('nan', 'none', 'nat'):
                    continue
                for part in [p.strip() for p in raw.split(';') if p.strip()]:
                    code = desc_to_code.get(part.lower())
                    if code and len(code) >= prefix_len:
                        prefix = code[:prefix_len]
                        all_prefixes_set.add(prefix)
                        e_code_counts[prefix] = e_code_counts.get(prefix, 0) + 1
            # Merge E-code counts into prefix_counts so the UI shows real values
            prefix_counts.update(e_code_counts)

    all_prefixes = sorted(all_prefixes_set)
    all_manufacturers = sorted(
        str(m) for m in df_with_dates[mfr_col].unique()
        if m is not None and str(m).strip() and str(m).strip().lower() not in ('nan', 'none', 'nat')
    )

    # Get level label for display
    level_label = LEVEL_CONFIG.get(level, {}).get('label', f'Level-{level}')

    return {
        "df_exploded": df_with_dates,
        "all_prefixes": all_prefixes,
        "all_manufacturers": all_manufacturers,
        "prefix_counts": prefix_counts,
        "date_col": date_col,
        "mfr_col": mfr_col,
        "total_rows": len(df),
        "rows_with_imdrf": len(df_exploded),
        "rows_with_dates": len(df_with_dates),
        "level": level,
        "level_label": level_label,
        "merge_log": merge_log,
    }


def _load_cleaned_dataframe(file_path: str) -> pd.DataFrame:
    """Load a cleaned MAUDE file into a DataFrame with string values."""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.xlsx', '.xls']:
        # keep_default_na=False prevents pandas from silently converting strings
        # like "NA", "N/A", "null" to float NaN — keeps them as literal strings.
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    elif file_ext == '.csv':
        df = pd.read_csv(file_path, dtype=str, encoding="utf-8",
                         on_bad_lines="skip", keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please upload CSV, XLS, or XLSX file.")

    for col in df.columns:
        # Force every cell to a plain Python str, then strip whitespace.
        # astype(str) converts residual float NaN → "nan"; replace removes it.
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "NaN": "", "None": "", "NaT": "", "<NA>": ""})

    return df


def get_imdrf_code_counts_all_levels(file_path, df: pd.DataFrame = None):
    """
    Get IMDRF code counts for Level-1, Level-2, and Level-3 from a cleaned file.
    This does NOT require a date column.

    Returns:
        dict: {1: {code: count}, 2: {code: count}, 3: {code: count}}
    """
    import os

    if df is None:
        df = _load_cleaned_dataframe(file_path)

    imdrf_col = find_imdrf_column(df)
    if imdrf_col is None:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(
            f"No IMDRF Code column found. This file does not appear to contain IMDRF codes. "
            f"Available columns: {available_cols}"
        )

    def count_for_level(level):
        prefixes = df[imdrf_col].apply(lambda x: extract_imdrf_prefixes(x, level=level))
        exploded = prefixes.explode()
        if exploded is None:
            return {}
        exploded = exploded.dropna()
        exploded = exploded[exploded.astype(str).str.strip() != ""]
        raw_counts = exploded.value_counts().to_dict()
        # Ensure all keys are plain strings — pandas 2.x can produce mixed-type
        # index values (float NaN alongside str) in object-dtype Series.
        # Also exclude any code whose Level-1 prefix (first 3 chars) is in
        # EXCLUDED_IMDRF_L1_PREFIXES (e.g. A24, A25).
        counts = {}
        for k, v in raw_counts.items():
            key = str(k).strip()
            if not key or key.lower() in ('nan', 'none', 'nat', ''):
                continue
            if key[:3].upper() in EXCLUDED_IMDRF_L1_PREFIXES:
                continue
            counts[key] = int(v)
        return counts

    return {
        1: count_for_level(1),
        2: count_for_level(2),
        3: count_for_level(3)
    }


def load_imdrf_code_descriptions(annex_file_path: str) -> Dict[int, Dict[str, str]]:
    """
    Load IMDRF code descriptions from the Annexes A-G consolidated file.

    Returns:
        dict: {1: {code: description}, 2: {code: description}, 3: {code: description}}
    """
    import pandas as pd

    if not annex_file_path or not os.path.exists(annex_file_path):
        raise ValueError("Annex file not found. Please upload the Annexes A-G consolidated file.")

    xl = pd.ExcelFile(annex_file_path)
    level_descriptions = {1: {}, 2: {}, 3: {}}

    required_cols = {"Level 1 Term", "Level 2 Term", "Level 3 Term", "Code"}

    for sheet in xl.sheet_names:
        df_raw = xl.parse(sheet_name=sheet, header=None, dtype=str)

        # Find header row
        header_row_idx = None
        for i in range(min(50, len(df_raw))):
            row_vals = [str(v).strip() for v in df_raw.iloc[i].tolist()]
            if "Level 1 Term" in row_vals:
                header_row_idx = i
                break
        if header_row_idx is None:
            continue

        df = xl.parse(sheet_name=sheet, header=header_row_idx, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]

        if not required_cols.issubset(set(df.columns)):
            continue

        # Forward fill
        df["Level 1 Term"] = df["Level 1 Term"].ffill()
        df["Level 2 Term"] = df["Level 2 Term"].ffill()

        for _, row in df.iterrows():
            code = "" if row.get("Code") is None else str(row.get("Code")).strip()
            if not code or code.lower() == "nan":
                continue

            if len(code) == 3:
                desc = str(row.get("Level 1 Term") or "").strip()
                if desc:
                    level_descriptions[1].setdefault(code, desc)
            elif len(code) == 5:
                desc = str(row.get("Level 2 Term") or "").strip()
                if desc:
                    level_descriptions[2].setdefault(code, desc)
            elif len(code) == 7:
                desc = str(row.get("Level 3 Term") or "").strip()
                if desc:
                    level_descriptions[3].setdefault(code, desc)

    return level_descriptions


def get_imdrf_code_counts_all_levels_with_descriptions(file_path: str, annex_file_path: str, df: pd.DataFrame = None):
    """
    Get IMDRF code counts for Level-1, Level-2, and Level-3 from a cleaned file,
    and attach the Annex description for each code.

    Returns:
        dict: {level: {code: {"count": int, "description": str}}}
    """
    counts_by_level = get_imdrf_code_counts_all_levels(file_path, df=df)
    descriptions_by_level = load_imdrf_code_descriptions(annex_file_path)

    merged = {}
    for level, counts in counts_by_level.items():
        level_desc = descriptions_by_level.get(level, {})
        merged[level] = {}
        for code, count in counts.items():
            merged[level][code] = {
                "count": count,
                "description": level_desc.get(code, "")
            }

    return merged


def get_patient_problem_counts(file_path: str, df: pd.DataFrame = None) -> Dict[str, int]:
    """
    Get patient problem counts from a cleaned file.

    Returns:
        dict: {patient_problem: count}
    """
    if df is None:
        df = _load_cleaned_dataframe(file_path)

    patient_col = None
    for col in df.columns:
        col_norm = str(col).strip().lower().replace('_', ' ')
        if col_norm in {"patient problem", "patient problems", "patient problem text"}:
            patient_col = col
            break
    if patient_col is None:
        for col in df.columns:
            col_norm = str(col).strip().lower().replace('_', ' ')
            if "patient" in col_norm and "problem" in col_norm:
                patient_col = col
                break

    if patient_col is None:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(
            f"No Patient Problem column found. Available columns: {available_cols}"
        )

    counts: Dict[str, int] = {}
    for value in df[patient_col].tolist():
        if value is None:
            continue
        raw = str(value).strip()
        if not raw or raw.lower() == "nan":
            continue
        parts = [p.strip() for p in raw.split(';') if p.strip()]
        if not parts:
            continue
        for part in parts:
            counts[part] = counts.get(part, 0) + 1

    return counts


def get_imdrf_code_monthly_counts(file_path: str, df: pd.DataFrame = None):
    """
    Get month-wise event counts for each IMDRF code at all levels.

    Months span the full date range of the data (including months with zero events).
    Counts are combined across all manufacturers.

    Returns:
        dict with:
        - 'months': sorted list of 'YYYY-MM' strings covering full date range
        - 'counts': {level: {code: {month_str: count}}}
          e.g. {1: {'A05': {'2023-01': 3, '2023-02': 0, ...}}, ...}
        Returns empty structure if date column not found.
    """
    if df is None:
        df = _load_cleaned_dataframe(file_path)

    imdrf_col = find_imdrf_column(df)
    if imdrf_col is None:
        return {'months': [], 'counts': {1: {}, 2: {}, 3: {}}}

    date_col = find_date_column(df)
    if date_col is None:
        return {'months': [], 'counts': {1: {}, 2: {}, 3: {}}}

    df = df.copy()
    df['_parsed_date'] = df[date_col].apply(parse_flexible_date)
    df_dated = df[df['_parsed_date'].notna()].copy()

    if df_dated.empty:
        return {'months': [], 'counts': {1: {}, 2: {}, 3: {}}}

    # 'YYYY-MM' string for each row
    df_dated['_month'] = df_dated['_parsed_date'].dt.to_period('M').astype(str)

    # Full month range including gaps
    all_periods = pd.period_range(
        start=df_dated['_parsed_date'].min(),
        end=df_dated['_parsed_date'].max(),
        freq='M'
    )
    all_months_str = [str(p) for p in all_periods]

    counts_by_level = {}
    for level in [1, 2, 3]:
        df_dated['_prefixes'] = df_dated[imdrf_col].apply(
            lambda x: extract_imdrf_prefixes(x, level=level)
        )
        df_exp = df_dated[df_dated['_prefixes'].apply(len) > 0].copy()
        df_exp = df_exp.explode('_prefixes').rename(columns={'_prefixes': '_code'})

        # Exclude A24/A25
        df_exp = df_exp[
            ~df_exp['_code'].astype(str).str[:3].str.upper().isin(EXCLUDED_IMDRF_L1_PREFIXES)
        ]

        if df_exp.empty:
            counts_by_level[level] = {}
            continue

        # Count per code per month
        grouped = df_exp.groupby(['_code', '_month']).size()
        level_counts = {}
        for (code, month), cnt in grouped.items():
            code = str(code).strip()
            if not code or code.lower() in ('nan', 'none', 'nat'):
                continue
            if code not in level_counts:
                level_counts[code] = {m: 0 for m in all_months_str}
            if month in level_counts[code]:
                level_counts[code][month] = int(cnt)

        counts_by_level[level] = level_counts

    return {'months': all_months_str, 'counts': counts_by_level}


def get_imdrf_code_manufacturer_monthly_counts(file_path: str, df: pd.DataFrame = None):
    """
    Get per-manufacturer, per-month event counts for every IMDRF code at all levels.

    Months span the full date range (including gaps filled with 0).
    Manufacturers with zero contribution to a code are excluded from that code's data.

    Returns:
        dict with:
        - 'months': sorted list of 'YYYY-MM' strings covering the full date range
        - 'data': {level: {code: {mfr: {month_str: count}}}}
    """
    if df is None:
        df = _load_cleaned_dataframe(file_path)

    imdrf_col = find_imdrf_column(df)
    if imdrf_col is None:
        return {'months': [], 'data': {1: {}, 2: {}, 3: {}}}

    date_col = find_date_column(df)
    if date_col is None:
        return {'months': [], 'data': {1: {}, 2: {}, 3: {}}}

    mfr_col = find_manufacturer_column(df)
    df = df.copy()
    if mfr_col is None:
        df['_manufacturer'] = 'All Data'
        mfr_col = '_manufacturer'

    df['_parsed_date'] = df[date_col].apply(parse_flexible_date)
    df_dated = df[df['_parsed_date'].notna()].copy()

    if df_dated.empty:
        return {'months': [], 'data': {1: {}, 2: {}, 3: {}}}

    df_dated['_month'] = df_dated['_parsed_date'].dt.to_period('M').astype(str)

    all_periods = pd.period_range(
        start=df_dated['_parsed_date'].min(),
        end=df_dated['_parsed_date'].max(),
        freq='M'
    )
    all_months_str = [str(p) for p in all_periods]

    data_by_level = {}
    for level in [1, 2, 3]:
        df_dated['_prefixes'] = df_dated[imdrf_col].apply(
            lambda x: extract_imdrf_prefixes(x, level=level)
        )
        df_exp = df_dated[df_dated['_prefixes'].apply(len) > 0].copy()
        df_exp = df_exp.explode('_prefixes').rename(columns={'_prefixes': '_code'})

        # Exclude A24/A25
        df_exp = df_exp[
            ~df_exp['_code'].astype(str).str[:3].str.upper().isin(EXCLUDED_IMDRF_L1_PREFIXES)
        ]

        if df_exp.empty:
            data_by_level[level] = {}
            continue

        grouped = df_exp.groupby(['_code', mfr_col, '_month']).size()

        level_data: dict = {}
        for (code, mfr, month), cnt in grouped.items():
            code = str(code).strip()
            mfr = str(mfr).strip()
            if not code or code.lower() in ('nan', 'none', 'nat'):
                continue
            if not mfr or mfr.lower() in ('nan', 'none', 'nat'):
                continue
            if code not in level_data:
                level_data[code] = {}
            if mfr not in level_data[code]:
                level_data[code][mfr] = {m: 0 for m in all_months_str}
            if month in level_data[code][mfr]:
                level_data[code][mfr][month] = int(cnt)

        data_by_level[level] = level_data

    return {'months': all_months_str, 'data': data_by_level}


def get_mfr_comparison_data(df_exploded, prefix, mfr_col, manufacturer):
    """
    Monthly comparison data for one manufacturer vs all others for a specific IMDRF prefix.

    Returns:
        {
            'months': ['YYYY-MM', ...],
            'selected_mfr': [int, ...] or None if manufacturer absent from data,
            'others_mean': [float, ...],
            'flat_avg': float,             # total prefix count / n active manufacturers
            'n_active': int,
            'selected_in_data': bool,
        }
    """
    df_prefix = df_exploded[df_exploded['imdrf_prefix'] == prefix].copy()

    if df_prefix.empty:
        return {
            'months': [], 'selected_mfr': None, 'others_mean': [],
            'flat_avg': 0.0, 'n_active': 0, 'selected_in_data': False,
        }

    mfr_counts = df_prefix[mfr_col].value_counts()
    active_mfrs = set(mfr_counts[mfr_counts > 0].index.tolist())
    n_active = len(active_mfrs)
    total_count = int(len(df_prefix))
    flat_avg = round(total_count / n_active, 4) if n_active > 0 else 0.0

    selected_in_data = manufacturer in active_mfrs
    other_active_mfrs = sorted(active_mfrs - {manufacturer})
    n_others = len(other_active_mfrs)

    df_prefix = df_prefix[df_prefix['parsed_date'].notna()].copy()
    df_prefix['_month'] = df_prefix['parsed_date'].dt.to_period('M')
    monthly_grouped = df_prefix.groupby([mfr_col, '_month']).size()
    all_months = sorted(df_prefix['_month'].dropna().unique())

    months_str, selected_counts, others_means = [], [], []
    for month in all_months:
        months_str.append(str(month))
        selected_counts.append(int(monthly_grouped.get((manufacturer, month), 0)))
        if n_others > 0:
            other_vals = [int(monthly_grouped.get((m, month), 0)) for m in other_active_mfrs]
            others_means.append(round(sum(other_vals) / n_others, 4))
        else:
            others_means.append(0.0)

    return {
        'months': months_str,
        'selected_mfr': selected_counts if selected_in_data else None,
        'others_mean': others_means,
        'flat_avg': flat_avg,
        'n_active': n_active,
        'selected_in_data': selected_in_data,
    }


def get_top_manufacturers_for_prefix(df_exploded, prefix, mfr_col, top_n=5):
    """
    Get top N manufacturers by volume for a specific IMDRF prefix.

    Args:
        df_exploded: DataFrame with exploded IMDRF prefixes
        prefix: IMDRF prefix to filter by
        mfr_col: Name of manufacturer column
        top_n: Number of top manufacturers to return

    Returns:
        list of top manufacturer names
    """
    df_prefix = df_exploded[df_exploded['imdrf_prefix'] == prefix]
    mfr_counts = df_prefix[mfr_col].value_counts()
    return mfr_counts.head(top_n).index.tolist()


# ---------------------------------------------------------------------------
# Patient Problem → IMDRF E-code mapping helpers
# ---------------------------------------------------------------------------

def _build_e_desc_to_code_map(annex_file_path: str):
    """
    Parse Annex E sheet and return two mappings:
      - desc_to_code: {normalized_description_lower: code}  (L1/L2/L3 terms → E code)
      - code_to_desc: {code: description}

    Only codes whose first char is 'E' are included.
    For duplicate descriptions, the first occurrence wins (setdefault).
    """
    desc_to_code: Dict[str, str] = {}
    code_to_desc: Dict[str, str] = {}

    if not annex_file_path or not os.path.exists(annex_file_path):
        return desc_to_code, code_to_desc

    xl = pd.ExcelFile(annex_file_path)

    for sheet in xl.sheet_names:
        if sheet.strip().upper() != 'E':
            continue

        df_raw = xl.parse(sheet_name=sheet, header=None, dtype=str)
        header_row_idx = None
        for i in range(min(50, len(df_raw))):
            row_vals = [str(v).strip() for v in df_raw.iloc[i].tolist()]
            if "Level 1 Term" in row_vals:
                header_row_idx = i
                break
        if header_row_idx is None:
            continue

        df_e = xl.parse(sheet_name=sheet, header=header_row_idx, dtype=str)
        df_e.columns = [str(c).strip() for c in df_e.columns]

        required = {"Level 1 Term", "Level 2 Term", "Level 3 Term", "Code"}
        if not required.issubset(set(df_e.columns)):
            continue

        df_e["Level 1 Term"] = df_e["Level 1 Term"].ffill()
        df_e["Level 2 Term"] = df_e["Level 2 Term"].ffill()

        for _, row in df_e.iterrows():
            raw_code = str(row.get("Code", "")).strip()
            if not raw_code or raw_code.lower() in ("nan", "none", ""):
                continue
            code = re.sub(r'[^A-Za-z0-9]', '', raw_code).upper()
            if not code.startswith('E'):
                continue

            if len(code) == 3:
                desc = str(row.get("Level 1 Term", "")).strip()
            elif len(code) == 5:
                desc = str(row.get("Level 2 Term", "")).strip()
            elif len(code) == 7:
                desc = str(row.get("Level 3 Term", "")).strip()
            else:
                continue

            if desc and desc.lower() not in ("nan", "none", ""):
                desc_to_code.setdefault(desc.lower(), code)
                code_to_desc.setdefault(code, desc)

    return desc_to_code, code_to_desc


def get_patient_problem_e_code_monthly_counts(file_path: str, annex_file_path: str,
                                               df: pd.DataFrame = None) -> Dict:
    """
    Map patient problem text descriptions to IMDRF E codes (Annex E) and return
    month-wise counts at Level-1, Level-2, and Level-3.

    Patient Problem column values are split on ';' and matched case-insensitively
    against term descriptions from Annex E. Unmatched values are skipped.

    Returns:
        dict with:
        - 'months': sorted list of 'YYYY-MM' strings covering full date range
        - 'counts': {level: {code: {month_str: count}}}
        - 'totals': {level: {code: {"count": int, "description": str}}}
    """
    if df is None:
        df = _load_cleaned_dataframe(file_path)

    desc_to_code, code_to_desc = _build_e_desc_to_code_map(annex_file_path)

    # Find patient problem column
    patient_col = None
    for col in df.columns:
        col_norm = str(col).strip().lower().replace('_', ' ')
        if col_norm in {"patient problem", "patient problems", "patient problem text"}:
            patient_col = col
            break
    if patient_col is None:
        for col in df.columns:
            col_norm = str(col).strip().lower().replace('_', ' ')
            if "patient" in col_norm and "problem" in col_norm:
                patient_col = col
                break

    if patient_col is None or not desc_to_code:
        return {'months': [], 'counts': {1: {}, 2: {}, 3: {}}, 'totals': {1: {}, 2: {}, 3: {}}}

    # Find date column
    date_col = find_date_column(df)

    df = df.copy()
    df['_parsed_date'] = df[date_col].apply(parse_flexible_date) if date_col else pd.NaT
    df_dated = df[df['_parsed_date'].notna()].copy() if date_col else df.copy()

    has_dates = date_col is not None and not df_dated.empty
    if has_dates:
        df_dated['_month'] = df_dated['_parsed_date'].dt.to_period('M').astype(str)
        all_periods = pd.period_range(
            start=df_dated['_parsed_date'].min(),
            end=df_dated['_parsed_date'].max(),
            freq='M'
        )
        all_months_str = [str(p) for p in all_periods]
    else:
        df_dated = df.copy()
        all_months_str = []

    # Explode patient problem column into rows: each part mapped to an E code
    rows = []
    for _, row in df_dated.iterrows():
        raw = str(row.get(patient_col, "")).strip()
        if not raw or raw.lower() in ('nan', 'none', 'nat', ''):
            continue
        month = row.get('_month', '') if has_dates else ''
        for part in [p.strip() for p in raw.split(';') if p.strip()]:
            code = desc_to_code.get(part.lower())
            if code:
                rows.append({'_code': code, '_month': month})

    if not rows:
        return {'months': all_months_str, 'counts': {1: {}, 2: {}, 3: {}}, 'totals': {1: {}, 2: {}, 3: {}}}

    df_exp = pd.DataFrame(rows)

    # Build counts and totals by level
    counts_by_level: Dict[int, Dict[str, Dict[str, int]]] = {}
    totals_by_level: Dict[int, Dict[str, Dict]] = {}

    for level in [1, 2, 3]:
        length = LEVEL_CONFIG[level]['length']
        df_exp['_prefix'] = df_exp['_code'].apply(
            lambda c: c[:length] if len(c) >= length else None
        )
        df_lv = df_exp[df_exp['_prefix'].notna()].copy()

        if df_lv.empty:
            counts_by_level[level] = {}
            totals_by_level[level] = {}
            continue

        # Totals
        total_counts = df_lv['_prefix'].value_counts().to_dict()
        totals_by_level[level] = {
            str(code): {
                "count": int(cnt),
                "description": code_to_desc.get(str(code), "")
            }
            for code, cnt in total_counts.items()
        }

        # Monthly breakdown
        if has_dates and '_month' in df_lv.columns:
            grouped = df_lv.groupby(['_prefix', '_month']).size()
            level_counts: Dict[str, Dict[str, int]] = {}
            for (code, month), cnt in grouped.items():
                code = str(code)
                if code not in level_counts:
                    level_counts[code] = {m: 0 for m in all_months_str}
                if month in level_counts[code]:
                    level_counts[code][month] = int(cnt)
            counts_by_level[level] = level_counts
        else:
            counts_by_level[level] = {}

    return {
        'months': all_months_str,
        'counts': counts_by_level,
        'totals': totals_by_level,
    }


def get_patient_problem_e_code_mfr_monthly_counts(file_path: str, annex_file_path: str,
                                                    df: pd.DataFrame = None) -> Dict:
    """
    Map patient problem text descriptions to IMDRF E codes and return per-manufacturer,
    per-month counts at Level-1, Level-2, and Level-3.

    Returns:
        dict with:
        - 'months': sorted list of 'YYYY-MM' strings covering the full date range
        - 'data': {level: {code: {mfr: {month_str: count}}}}
    """
    if df is None:
        df = _load_cleaned_dataframe(file_path)

    desc_to_code, code_to_desc = _build_e_desc_to_code_map(annex_file_path)

    # Find patient problem column
    patient_col = None
    for col in df.columns:
        col_norm = str(col).strip().lower().replace('_', ' ')
        if col_norm in {"patient problem", "patient problems", "patient problem text"}:
            patient_col = col
            break
    if patient_col is None:
        for col in df.columns:
            col_norm = str(col).strip().lower().replace('_', ' ')
            if "patient" in col_norm and "problem" in col_norm:
                patient_col = col
                break

    if patient_col is None or not desc_to_code:
        return {'months': [], 'data': {1: {}, 2: {}, 3: {}}}

    date_col = find_date_column(df)
    mfr_col = find_manufacturer_column(df)
    df = df.copy()
    if mfr_col is None:
        df['_manufacturer'] = 'All Data'
        mfr_col = '_manufacturer'

    if date_col is None:
        return {'months': [], 'data': {1: {}, 2: {}, 3: {}}}

    df['_parsed_date'] = df[date_col].apply(parse_flexible_date)
    df_dated = df[df['_parsed_date'].notna()].copy()

    if df_dated.empty:
        return {'months': [], 'data': {1: {}, 2: {}, 3: {}}}

    df_dated['_month'] = df_dated['_parsed_date'].dt.to_period('M').astype(str)

    all_periods = pd.period_range(
        start=df_dated['_parsed_date'].min(),
        end=df_dated['_parsed_date'].max(),
        freq='M'
    )
    all_months_str = [str(p) for p in all_periods]

    # Explode patient problems into rows with manufacturer + month
    rows = []
    for _, row in df_dated.iterrows():
        raw = str(row.get(patient_col, "")).strip()
        if not raw or raw.lower() in ('nan', 'none', 'nat', ''):
            continue
        mfr = str(row.get(mfr_col, "")).strip()
        month = str(row.get('_month', ''))
        for part in [p.strip() for p in raw.split(';') if p.strip()]:
            code = desc_to_code.get(part.lower())
            if code:
                rows.append({'_code': code, '_mfr': mfr, '_month': month})

    if not rows:
        return {'months': all_months_str, 'data': {1: {}, 2: {}, 3: {}}}

    df_exp = pd.DataFrame(rows)

    data_by_level: Dict[int, Dict] = {}
    for level in [1, 2, 3]:
        length = LEVEL_CONFIG[level]['length']
        df_exp['_prefix'] = df_exp['_code'].apply(
            lambda c: c[:length] if len(c) >= length else None
        )
        df_lv = df_exp[df_exp['_prefix'].notna()].copy()

        if df_lv.empty:
            data_by_level[level] = {}
            continue

        grouped = df_lv.groupby(['_prefix', '_mfr', '_month']).size()
        level_data: Dict = {}
        for (code, mfr, month), cnt in grouped.items():
            code = str(code)
            mfr = str(mfr)
            if not mfr or mfr.lower() in ('nan', 'none', 'nat'):
                continue
            if code not in level_data:
                level_data[code] = {}
            if mfr not in level_data[code]:
                level_data[code][mfr] = {m: 0 for m in all_months_str}
            if month in level_data[code][mfr]:
                level_data[code][mfr][month] = int(cnt)

        data_by_level[level] = level_data

    return {'months': all_months_str, 'data': data_by_level}


# ---------------------------------------------------------------------------
# PDF Report generation helpers
# ---------------------------------------------------------------------------

def compute_report_data(df_current, df_hist, mfr_col, manufacturer,
                        code_filter, period_from, period_to, grain, level):
    """
    Compute all data needed for the IMDRF Trend Analysis PDF report.

    Args:
        df_current: df_exploded for user's selected period (has imdrf_prefix, parsed_date)
        df_hist:    df_exploded for preceding 2-year historical period
        mfr_col:    manufacturer column name
        manufacturer: selected manufacturer string
        code_filter: IMDRF code string OR "ALL" (→ top 5)
        period_from, period_to: "YYYY-MM-DD" strings
        grain:  "M" (monthly) or "Q" (quarterly)
        level:  1, 2, or 3 (for display label)

    Returns:
        dict with keys: manufacturer, period_from, period_to, hist_from, hist_to,
        mfr_events, grand_total, level_label, grain_label,
        top5 (list of dicts), historical (list of dicts),
        trends (dict of code → {code, labels, mfr_values, peers_values})
    """
    period_from_ts = pd.Timestamp(period_from)
    period_to_ts   = pd.Timestamp(period_to)
    grain_label    = 'Monthly' if grain == 'M' else 'Quarterly'
    level_label    = LEVEL_CONFIG.get(level, {}).get('label', f'Level-{level}')

    # ── Filter current period ──────────────────────────────────────────────
    df_period = df_current[
        (df_current['parsed_date'] >= period_from_ts) &
        (df_current['parsed_date'] <= period_to_ts)
    ].copy()

    # Exclude A24/A25
    df_period = df_period[
        ~df_period['imdrf_prefix'].astype(str).str[:3].str.upper().isin(EXCLUDED_IMDRF_L1_PREFIXES)
    ]

    grand_total = len(df_period)
    df_mfr = df_period[df_period[mfr_col] == manufacturer]
    mfr_total  = len(df_mfr)

    # ── Top codes for selected manufacturer ───────────────────────────────
    if mfr_total > 0:
        mfr_code_counts = df_mfr['imdrf_prefix'].value_counts()
    else:
        mfr_code_counts = pd.Series(dtype=int)

    if code_filter and code_filter.upper() != 'ALL':
        codes_to_show = [code_filter]
    else:
        codes_to_show = mfr_code_counts.head(5).index.tolist()
        if not codes_to_show:
            # Fallback: top codes across all manufacturers
            codes_to_show = df_period['imdrf_prefix'].value_counts().head(5).index.tolist()

    top5 = []
    for code in codes_to_show:
        cnt = int(mfr_code_counts.get(code, 0))
        prop = cnt / mfr_total if mfr_total > 0 else 0.0
        top5.append({'code': code, 'mfr_count': cnt, 'proportion': prop})

    # ── Historical proportions ─────────────────────────────────────────────
    df_hist_clean = df_hist[
        ~df_hist['imdrf_prefix'].astype(str).str[:3].str.upper().isin(EXCLUDED_IMDRF_L1_PREFIXES)
    ]
    hist_grand_total = len(df_hist_clean)

    # Derive hist date range from actual data
    if not df_hist_clean.empty and df_hist_clean['parsed_date'].notna().any():
        hist_from = df_hist_clean['parsed_date'].min().strftime('%Y-%m-%d')
        hist_to   = df_hist_clean['parsed_date'].max().strftime('%Y-%m-%d')
    else:
        hist_from = ''
        hist_to   = ''

    historical = []
    for code in codes_to_show:
        hist_cnt  = int((df_hist_clean['imdrf_prefix'] == code).sum())
        hist_prop = hist_cnt / hist_grand_total if hist_grand_total > 0 else 0.0
        historical.append({'code': code, 'hist_total': hist_cnt, 'hist_proportion': hist_prop})

    # ── Current period proportions (all manufacturers) ────────────────────
    current_period = []
    for code in codes_to_show:
        period_cnt  = int((df_period['imdrf_prefix'] == code).sum())
        period_prop = period_cnt / grand_total if grand_total > 0 else 0.0
        current_period.append({'code': code, 'period_total': period_cnt, 'period_proportion': period_prop})

    # ── Trend series per code ─────────────────────────────────────────────
    freq = 'M' if grain == 'M' else 'Q'
    full_range = pd.period_range(start=period_from_ts, end=period_to_ts, freq=freq)
    labels = [str(p) for p in full_range]

    trends = {}
    for code in codes_to_show:
        df_code = df_period[df_period['imdrf_prefix'] == code].copy()
        df_code['_period'] = df_code['parsed_date'].dt.to_period(freq)

        mfr_df    = df_code[df_code[mfr_col] == manufacturer]
        peers_df  = df_code[df_code[mfr_col] != manufacturer]

        mfr_series   = mfr_df.groupby('_period').size().reindex(full_range, fill_value=0)
        peers_series = peers_df.groupby('_period').size().reindex(full_range, fill_value=0)

        trends[code] = {
            'code':        code,
            'labels':      labels,
            'mfr_values':  mfr_series.tolist(),
            'peers_values': peers_series.tolist(),
        }

    return {
        'manufacturer': manufacturer,
        'period_from':  period_from,
        'period_to':    period_to,
        'hist_from':    hist_from,
        'hist_to':      hist_to,
        'mfr_events':   mfr_total,
        'grand_total':  grand_total,
        'level_label':  level_label,
        'grain_label':  grain_label,
        'top5':            top5,
        'historical':      historical,
        'current_period':  current_period,
        'trends':          trends,
    }


def render_trend_chart(code, labels, mfr_values, peers_values, manufacturer, grain_label):
    """
    Generate a matplotlib trend chart PNG and return as BytesIO.

    Two lines: selected manufacturer (blue) and All Others/Peers (amber).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(labels, mfr_values,   color='#2563eb', linewidth=2,
            marker='o', markersize=4, label=manufacturer)
    ax.plot(labels, peers_values, color='#f59e0b', linewidth=2,
            marker='s', markersize=4, label='All Others (Peers)')
    ax.set_title(f'IMDRF Code {code} — {grain_label} Trend', fontsize=11, fontweight='bold')
    ax.set_xlabel('Period', fontsize=9)
    ax.set_ylabel('Event Count', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Show a subset of x-axis labels to avoid overlap
    n = len(labels)
    if n > 24:
        step = max(1, n // 12)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([labels[i] for i in range(0, n, step)],
                           rotation=45, ha='right', fontsize=8)
    else:
        plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def compute_proportions(df_current, df_hist, code, period_from, period_to):
    """
    Compute proportion statistics for a specific IMDRF code across combined
    historical + current period data (all manufacturers).

    Args:
        df_current: df_exploded for user's selected period
        df_hist:    df_exploded for preceding 2-year historical period
        code:       IMDRF code string (e.g., "A05")
        period_from, period_to: "YYYY-MM-DD" strings (user's specified range)

    Returns:
        {
            'total_proportion': float,
            'total_code_count': int,
            'total_all_count': int,
            'period_months': ['YYYY-MM', ...],
            'monthly': {'YYYY-MM': {'code_count': int, 'total_count': int, 'proportion': float}}
        }
    """
    # Combine both datasets (hist + current)
    df_combined = pd.concat([df_hist, df_current], ignore_index=True)

    # Exclude A24/A25
    df_combined = df_combined[
        ~df_combined['imdrf_prefix'].astype(str).str[:3].str.upper().isin(EXCLUDED_IMDRF_L1_PREFIXES)
    ]

    total_all_count  = len(df_combined)
    total_code_count = int((df_combined['imdrf_prefix'] == code).sum())
    total_proportion = total_code_count / total_all_count if total_all_count > 0 else 0.0

    # Monthly proportions for user's specified period only
    period_from_ts = pd.Timestamp(period_from)
    period_to_ts   = pd.Timestamp(period_to)

    df_period = df_combined[
        (df_combined['parsed_date'] >= period_from_ts) &
        (df_combined['parsed_date'] <= period_to_ts)
    ].copy()

    period_months = []
    monthly = {}

    if not df_period.empty:
        df_period['_month'] = df_period['parsed_date'].dt.to_period('M').astype(str)
        full_range   = pd.period_range(start=period_from_ts, end=period_to_ts, freq='M')
        period_months = [str(p) for p in full_range]

        for month in period_months:
            df_month   = df_period[df_period['_month'] == month]
            month_total = len(df_month)
            month_code  = int((df_month['imdrf_prefix'] == code).sum())
            monthly[month] = {
                'code_count':  month_code,
                'total_count': month_total,
                'proportion':  month_code / month_total if month_total > 0 else 0.0,
            }

    return {
        'total_proportion': total_proportion,
        'total_code_count': total_code_count,
        'total_all_count':  total_all_count,
        'period_months':    period_months,
        'monthly':          monthly,
    }


def get_hist_code_table(df_hist):
    """
    Compute a code distribution table from the historical (2-year) dataset.

    For each IMDRF prefix found in df_hist, returns its event count and
    proportion relative to the total events in the dataset.

    Args:
        df_hist: df_exploded produced by prepare_data_for_insights()

    Returns:
        {
            'total_events': int,
            'rows': [
                {'code': str, 'count': int, 'proportion': float},
                ...
            ]  -- sorted by count descending
        }
    """
    # Exclude A24/A25
    df = df_hist[
        ~df_hist['imdrf_prefix'].astype(str).str[:3].str.upper().isin(EXCLUDED_IMDRF_L1_PREFIXES)
    ].copy()

    total_events = len(df)
    if total_events == 0:
        return {'total_events': 0, 'rows': []}

    counts = df['imdrf_prefix'].value_counts()
    rows = [
        {
            'code':       code,
            'count':      int(cnt),
            'proportion': round(int(cnt) / total_events, 6),
        }
        for code, cnt in counts.items()
    ]
    return {'total_events': total_events, 'rows': rows}


def build_report_pdf(report_data, chart_images):
    """
    Build the IMDRF Trend Analysis PDF using ReportLab.

    Args:
        report_data: dict returned by compute_report_data()
        chart_images: list of BytesIO PNG images (one per code in trends)

    Returns:
        bytes — raw PDF content
    """
    from io import BytesIO
    from datetime import datetime as dt

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image, HRFlowable)
    from reportlab.lib.units import inch, cm

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    page_width = A4[0] - 4*cm  # usable width

    styles   = getSampleStyleSheet()
    navy     = colors.HexColor('#1e3a5f')
    light_bg = colors.HexColor('#f0f4ff')
    alt_bg   = colors.HexColor('#e8f0fe')

    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'],
                                 fontSize=18, textColor=navy, spaceAfter=4)
    sub_style   = ParagraphStyle('ReportSub', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.grey, spaceAfter=12)
    h2_style    = ParagraphStyle('H2', parent=styles['Heading2'],
                                 fontSize=13, textColor=navy, spaceBefore=16, spaceAfter=8)
    body_style  = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, leading=15, spaceAfter=10)

    def make_table(data_rows, col_widths, header_row):
        table_data = [header_row] + data_rows
        tbl = Table(table_data, colWidths=col_widths)
        style = TableStyle([
            ('BACKGROUND',  (0, 0), (-1, 0), navy),
            ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
            ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',    (0, 0), (-1, 0), 10),
            ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN',       (0, 1), (0, -1), 'LEFT'),
            ('FONTSIZE',    (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
            ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#c7d2e8')),
            ('TOPPADDING',  (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ])
        tbl.setStyle(style)
        return tbl

    story = []

    # ── Cover / Title ──────────────────────────────────────────────────────
    story.append(Paragraph('IMDRF Trend Analysis Report', title_style))
    story.append(Paragraph(f"Generated: {dt.now().strftime('%d %B %Y, %H:%M')}", sub_style))
    story.append(HRFlowable(width='100%', thickness=1.5, color=navy, spaceAfter=16))

    # ── Section 1: Overview ────────────────────────────────────────────────
    story.append(Paragraph('1. Overview', h2_style))

    mfr       = report_data['manufacturer']
    pfrom     = report_data['period_from']
    pto       = report_data['period_to']
    mfr_evts  = report_data['mfr_events']
    total     = report_data['grand_total']
    lvl_lbl   = report_data['level_label']
    grain_lbl = report_data['grain_label']

    narrative = (
        f"The manufacturer <b>{mfr}</b> was selected for analysis for the period "
        f"<b>{pfrom}</b> to <b>{pto}</b>. The number of events identified for this "
        f"manufacturer is <b>{mfr_evts:,}</b> and <b>{total:,}</b> events overall "
        f"for all manufacturers. The top five events and their codes are as follows:"
    )
    story.append(Paragraph(narrative, body_style))

    if report_data['top5']:
        col_w = [page_width * 0.25, page_width * 0.4, page_width * 0.35]
        header = ['Code', 'Events (Manufacturer)', 'Proportion of Mfr Events']
        rows = [
            [e['code'],
             str(e['mfr_count']),
             f"{e['proportion']*100:.1f}%"]
            for e in report_data['top5']
        ]
        story.append(make_table(rows, col_w, header))
    story.append(Spacer(1, 12))

    # ── Section 2: Historical Baseline ────────────────────────────────────
    story.append(Paragraph('2. Historical Baseline', h2_style))

    hist_from = report_data['hist_from']
    hist_to   = report_data['hist_to']
    hist_intro = (
        f"Using historical data of the preceding 2 years "
        f"(<b>{hist_from}</b> to <b>{hist_to}</b>), "
        f"the average proportion of events for the top five codes are calculated. "
        f"The results are as follows:"
    )
    story.append(Paragraph(hist_intro, body_style))

    if report_data['historical']:
        col_w = [page_width * 0.25, page_width * 0.4, page_width * 0.35]
        header = ['Code', 'Total Events (2-yr, All Mfrs)', 'Average Proportion (All Events)']
        rows = [
            [e['code'],
             str(e['hist_total']),
             f"{e['hist_proportion']*100:.2f}%"]
            for e in report_data['historical']
        ]
        story.append(make_table(rows, col_w, header))
    story.append(Spacer(1, 12))

    # ── Current period table (all manufacturers) ──────────────────────────
    story.append(Paragraph('Selected Period Distribution (All Manufacturers)', h2_style))
    period_intro = (
        f"The table below shows the total event counts and proportion for each code "
        f"across <b>all manufacturers</b> during the selected period "
        f"<b>{pfrom}</b> to <b>{pto}</b>."
    )
    story.append(Paragraph(period_intro, body_style))
    if report_data.get('current_period'):
        col_w = [page_width * 0.25, page_width * 0.4, page_width * 0.35]
        header = ['Code', f'Total Events ({pfrom} – {pto}, All Mfrs)', 'Proportion (All Events)']
        rows = [
            [e['code'],
             str(e['period_total']),
             f"{e['period_proportion']*100:.2f}%"]
            for e in report_data['current_period']
        ]
        story.append(make_table(rows, col_w, header))
    story.append(Spacer(1, 12))

    # ── Section 3: Trend Charts ────────────────────────────────────────────
    story.append(Paragraph('3. Trend Analysis', h2_style))

    trend_intro = (
        f"An analysis of the trend was done on a <b>{grain_lbl.lower()}</b> basis "
        f"for the period <b>{pfrom}</b> to <b>{pto}</b>. "
        f"The graphs for each code for this manufacturer vs peers are given below:"
    )
    story.append(Paragraph(trend_intro, body_style))

    img_width = page_width
    for buf_img in chart_images:
        img = Image(buf_img, width=img_width, height=img_width * 0.39)
        story.append(img)
        story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return buf.read()
