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


# ── PSUR-Style Detailed Report ────────────────────────────────────────────────

def compute_detailed_report_data(df_exploded, mfr_col, file_path, annex_path,
                                  year_from, year_to, level=1):
    """
    Compute all data needed for the PSUR-style detailed PDF report.

    Args:
        df_exploded:  DataFrame from prepare_data_for_insights
                      (columns: imdrf_prefix, parsed_date, mfr_col, ...)
        mfr_col:      manufacturer column name
        file_path:    path to cleaned data file (for patient problems / event types)
        annex_path:   path to IMDRF Annexes file (for descriptions)
        year_from, year_to: integer year bounds (inclusive)
        level:        analysis level (1/2/3), used for display label only

    Returns:
        dict with keys: year_from, year_to, years, total_events,
        total_manufacturers, level_label, top5_families, patient_problems,
        event_type_counts
    """
    # Filter df_exploded to year range (used for yearly counts + manufacturers)
    df = df_exploded[
        df_exploded['parsed_date'].notna() &
        (df_exploded['parsed_date'].dt.year >= year_from) &
        (df_exploded['parsed_date'].dt.year <= year_to)
    ].copy()

    df['_l1'] = df['imdrf_prefix'].str[:3].str.upper()
    years = list(range(year_from, year_to + 1))
    total_events = len(df)  # preliminary; updated below after dedup
    total_manufacturers = int(df[mfr_col].nunique()) if mfr_col in df.columns else 0

    # Load IMDRF descriptions  {1: {code: str}, 2: {code: str}, 3: {code: str}}
    try:
        descs = load_imdrf_code_descriptions(annex_path)
    except Exception:
        descs = {}
    l1_descs = descs.get(1, {})
    l2_descs = descs.get(2, {})
    l3_descs = descs.get(3, {})

    # ── Family grand totals and L1/L2/L3 hierarchy ────────────────────────
    # The cleaned XLSX is doubly-exploded (device problem × patient problem),
    # so raw counts on it inflate every figure. We dedupe by Report Number
    # first to recover one row per actual MAUDE event, then extract codes
    # at each level from the original pipe-separated IMDRF Code column.
    # This both fixes the Grand Total (sum-of-rows ≤ family total because
    # L2/L3 are sub-views of L1) and restores L2/L3 hierarchy visibility.
    raw_df = pd.DataFrame()
    raw_dedup = pd.DataFrame()
    patient_problems: dict = {}
    event_type_counts: dict = {}
    l1_all: dict = {}
    l2_all: dict = {}
    l3_all: dict = {}

    family_grand_totals: dict = {
        str(k): int(v) for k, v in df['_l1'].value_counts().items()
        if str(k).strip() and str(k).strip().lower() not in ('nan', 'none', '')
    }

    try:
        raw_df = _load_cleaned_dataframe(file_path)
        date_col_raw = find_date_column(raw_df)
        if date_col_raw:
            raw_df['_pd'] = raw_df[date_col_raw].apply(parse_flexible_date)
            raw_df = raw_df[
                raw_df['_pd'].notna() &
                (raw_df['_pd'].dt.year >= year_from) &
                (raw_df['_pd'].dt.year <= year_to)
            ].copy()

        # Locate a Report Number column to dedupe on (column names vary by
        # source: "Report Number", "report_number", "MDR Report Key", etc.).
        report_col = None
        for _c in raw_df.columns:
            _cn = str(_c).strip().lower().replace('_', ' ')
            if _cn in ('report number', 'mdr report key', 'mdr_report_key', 'reportnumber'):
                report_col = _c
                break
        if report_col is not None:
            raw_dedup = raw_df.drop_duplicates(subset=[report_col]).copy()
        else:
            # Fallback: dedupe on (IMDRF Code + date_col) which keeps one row
            # per (event, code-set) pair — still removes patient-problem dupes.
            _imdrf_c = find_imdrf_column(raw_df)
            _key_cols = [c for c in [_imdrf_c, date_col_raw] if c]
            raw_dedup = raw_df.drop_duplicates(subset=_key_cols).copy() if _key_cols else raw_df.copy()

        _all_counts = get_imdrf_code_counts_all_levels(file_path=None, df=raw_dedup)
        l1_all = _all_counts.get(1, {})
        l2_all = _all_counts.get(2, {})
        l3_all = _all_counts.get(3, {})

        # Patient problems still come from the doubly-exploded raw_df —
        # each (event × patient problem) row is the correct unit of count
        # for the patient-problem distribution.
        try:
            patient_problems = get_patient_problem_counts(file_path=None, df=raw_df)
        except Exception:
            pass
        # Event-type counts use the deduped frame so each event counts once.
        for _col in raw_dedup.columns:
            if _col.strip().lower() in ('event type', 'event_type'):
                event_type_counts = {
                    str(k): int(v)
                    for k, v in raw_dedup[_col].value_counts().items()
                    if str(k).strip() and str(k).strip().lower() not in ('nan', 'none', '')
                }
                break

        # Update total_events to the actual unique MAUDE event count (deduped
        # by report number), rather than exploded IMDRF-code row count.
        if len(raw_dedup) > 0:
            total_events = len(raw_dedup)
    except Exception:
        pass

    # ── Determine top-5 A-code families ──────────────────────────────────────
    # Prefer all-level grand totals; fall back to df_exploded l1_counts
    if family_grand_totals:
        a_families = {k: v for k, v in family_grand_totals.items()
                      if not k.startswith('E') and k not in EXCLUDED_IMDRF_L1_PREFIXES}
        top5_codes = sorted(a_families, key=lambda c: -a_families[c])[:5]
    else:
        l1_counts = df['_l1'].value_counts()
        a_codes = [c for c in l1_counts.index if not c.startswith('E')]
        top5_codes = a_codes[:5]

    top5_families = []
    for code in top5_codes:
        df_code = df[df['_l1'] == code]

        # Year-by-year totals (from df_exploded — used for the mini year table)
        yearly = [int((df_code['parsed_date'].dt.year == yr).sum()) for yr in years]

        # Top manufacturers (from df_exploded)
        top_mfrs = []
        if mfr_col in df_code.columns:
            _fam_gt = family_grand_totals.get(code, 0) or int(l1_counts.get(code, 0) if not family_grand_totals else 0)
            for k, v in df_code[mfr_col].value_counts().head(5).items():
                top_mfrs.append({'name': str(k), 'count': int(v)})

        top_l2 = []
        for k, v in df_code['imdrf_prefix'].str[:5].value_counts().head(5).items():
            k_str = str(k)
            if len(k_str) >= 5:
                top_l2.append({
                    'code': k_str,
                    'count': int(v),
                    'description': str(l2_descs.get(k_str, '')),
                })

        # ── Full L1/L2/L3 hierarchy ───────────────────────────────────────────
        # Built from all-level counts so every code at its true granularity
        # appears exactly once. Grand total = sum of all rows = total events.
        full_hierarchy = []

        # L1 direct row (events coded exactly as the 3-char L1 code)
        l1_direct = l1_all.get(code, 0)
        full_hierarchy.append({
            'code': code,
            'level': 'Level 1',
            'description': str(l1_descs.get(code, '')),
            'total': l1_direct,
        })

        # L2 rows, sorted; under each L2, its L3 children sorted
        l2_codes = sorted(k for k in l2_all if k[:3] == code)
        for l2c in l2_codes:
            full_hierarchy.append({
                'code': l2c,
                'level': 'Level 2',
                'description': str(l2_descs.get(l2c, '')),
                'total': l2_all[l2c],
            })
            l3_codes = sorted(k for k in l3_all if k[:5] == l2c)
            for l3c in l3_codes:
                full_hierarchy.append({
                    'code': l3c,
                    'level': 'Level 3',
                    'description': str(l3_descs.get(l3c, '')),
                    'total': l3_all[l3c],
                })

        # Orphan L3 codes not under any L2 we listed (edge case)
        placed = {r['code'] for r in full_hierarchy}
        for l3c in sorted(k for k in l3_all if k[:3] == code and k not in placed):
            full_hierarchy.append({
                'code': l3c,
                'level': 'Level 3',
                'description': str(l3_descs.get(l3c, '')),
                'total': l3_all[l3c],
            })

        # Grand Total = sum of displayed rows so the table is internally
        # self-consistent (user-visible sanity check). Note: this can exceed
        # family_grand_totals[code] when an event lists both an L1 code and
        # its L2/L3 descendants in the same IMDRF Code field.
        grand_total = sum(r['total'] for r in full_hierarchy)

        top5_families.append({
            'code': code,
            'description': str(l1_descs.get(code, '')),
            'grand_total': grand_total,
            'family_total': int(family_grand_totals.get(code, 0)),
            'yearly_counts': yearly,
            'top_manufacturers': top_mfrs,
            'top_l2_codes': top_l2,
            'full_hierarchy': full_hierarchy,
        })

    return {
        'year_from': year_from,
        'year_to': year_to,
        'years': years,
        'total_events': total_events,
        'total_manufacturers': total_manufacturers,
        'level_label': LEVEL_CONFIG.get(level, {}).get('label', f'Level-{level}'),
        'top5_families': top5_families,
        'patient_problems': patient_problems,
        'event_type_counts': event_type_counts,
    }


def render_detailed_yoy_bar_chart(years, top5_families):
    """Grouped bar chart — total events per year for each top code family."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    n_fam = len(top5_families)
    n_yr  = len(years)
    x     = np.arange(n_yr)
    width = 0.7 / max(n_fam, 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, fam in enumerate(top5_families):
        offset = (i - n_fam / 2.0 + 0.5) * width
        label  = f"{fam['code']} — {fam['description'][:35]}" if fam['description'] else fam['code']
        ax.bar(x + offset, fam['yearly_counts'], width,
               label=label, color=palette[i % len(palette)], alpha=0.85)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Total Events', fontsize=11)
    ax.set_title('Year-over-Year Event Trends — Top Code Families',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.legend(fontsize=8, loc='upper left', framealpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def render_detailed_total_bar_chart(top5_families):
    """Horizontal bar — grand total events per code family."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO

    if not top5_families:
        return None

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    labels  = [
        f"{f['code']} — {f['description'][:38]}" if f['description'] else f['code']
        for f in top5_families
    ]
    totals  = [f['grand_total'] for f in top5_families]
    max_val = max(totals) if totals else 1

    fig, ax = plt.subplots(figsize=(10, max(3, len(top5_families) * 0.8)))
    bars = ax.barh(
        range(len(labels)),
        totals[::-1],
        color=[palette[i % len(palette)] for i in range(len(labels))],
        alpha=0.85,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel('Total Events', fontsize=11)
    ax.set_title('Total Events by Code Family (All Years)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, totals[::-1]):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def render_detailed_patient_problems_bar(pp_counts, top_n=10):
    """Horizontal bar — top N patient-reported problems by count."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO

    sorted_pp = sorted(pp_counts.items(), key=lambda x: -x[1])[:top_n]
    if not sorted_pp:
        return None

    labels = [p[:55] for p, _ in sorted_pp]
    counts = [c for _, c in sorted_pp]

    fig, ax = plt.subplots(figsize=(10, max(3, top_n * 0.55)))
    ax.barh(range(len(labels)), counts[::-1], color='#4e79a7', alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel('Number of Reports', fontsize=11)
    ax.set_title('Top Patient-Reported Problems', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def build_detailed_report_pdf(report_data, chart_images):
    """
    Build the PSUR-style detailed IMDRF analysis PDF using ReportLab.

    Args:
        report_data:   dict returned by compute_detailed_report_data()
        chart_images:  dict with optional BytesIO keys:
                       'yoy_bar', 'total_bar', 'patient_problems_bar'

    Returns:
        bytes — raw PDF content
    """
    from io import BytesIO
    from datetime import datetime as dt

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image, HRFlowable,
                                    PageBreak)
    from reportlab.lib.units import cm

    buf      = BytesIO()
    doc      = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    page_w   = A4[0] - 4*cm   # usable width ~17.0 cm

    styles   = getSampleStyleSheet()
    navy     = colors.HexColor('#1e3a5f')
    teal     = colors.HexColor('#0d6e6e')
    light_bg = colors.HexColor('#f0f4ff')
    mid_bg   = colors.HexColor('#dce8f8')
    accent   = colors.HexColor('#e8f5e9')

    title_style   = ParagraphStyle('DRTitle', parent=styles['Title'],
                                   fontSize=20, textColor=navy, spaceAfter=4, leading=26)
    sub_style     = ParagraphStyle('DRSub',   parent=styles['Normal'],
                                   fontSize=10, textColor=colors.grey, spaceAfter=6)
    h1_style      = ParagraphStyle('DRH1',    parent=styles['Heading1'],
                                   fontSize=15, textColor=navy, spaceBefore=18, spaceAfter=8,
                                   borderPad=4)
    h2_style      = ParagraphStyle('DRH2',    parent=styles['Heading2'],
                                   fontSize=12, textColor=teal, spaceBefore=14, spaceAfter=6)
    h3_style      = ParagraphStyle('DRH3',    parent=styles['Heading3'],
                                   fontSize=11, textColor=navy, spaceBefore=10, spaceAfter=4)
    body_style    = ParagraphStyle('DRBody',  parent=styles['Normal'],
                                   fontSize=10, leading=15, spaceAfter=8)
    note_style    = ParagraphStyle('DRNote',  parent=styles['Normal'],
                                   fontSize=9,  textColor=colors.grey, leading=13, spaceAfter=6)
    caption_style = ParagraphStyle('DRCaption', parent=styles['Normal'],
                                   fontSize=9, textColor=colors.grey, alignment=1, spaceAfter=10)

    def make_table(header_row, data_rows, col_widths, alt_color=light_bg):
        table_data = [header_row] + data_rows
        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), navy),
            ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
            ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1, 0), 9),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN',         (0, 1), (0, -1), 'LEFT'),
            ('ALIGN',         (1, 1), (1, -1), 'LEFT'),
            ('FONTSIZE',      (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, alt_color]),
            ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor('#c7d2e8')),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ]))
        return tbl

    def embed_image(buf_img, width=None, height=None):
        if buf_img is None:
            return None
        if hasattr(buf_img, 'seek'):
            buf_img.seek(0)
        w = width or page_w
        h = height or (w * 0.45)
        return Image(buf_img, width=w, height=h)

    story = []
    yr_from = report_data['year_from']
    yr_to   = report_data['year_to']
    years   = report_data['years']
    total   = report_data['total_events']
    n_mfrs  = report_data['total_manufacturers']
    fams    = report_data['top5_families']
    pp      = report_data['patient_problems']
    et      = report_data['event_type_counts']
    pi      = report_data.get('product_info') or {}

    pc_code        = pi.get('product_code', '')
    pc_name        = pi.get('device_name', '')
    pc_specialty   = pi.get('medical_specialty', '')
    pc_class       = pi.get('device_class', '')
    pc_reg         = pi.get('regulation_number', '')
    pc_definition  = pi.get('definition', '')
    pc_date_from   = pi.get('date_from', '')
    pc_date_to     = pi.get('date_to', '')

    # ── Cover Page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph('FDA MAUDE — Analysis Report', title_style))
    story.append(Paragraph(
        f"Generated: <b>{dt.now().strftime('%d %B %Y')}</b>",
        sub_style,
    ))
    story.append(HRFlowable(width='100%', thickness=2, color=navy, spaceAfter=14))

    # ── Product Code information block ────────────────────────────────────────
    if pc_code:
        prod_label_style = ParagraphStyle('ProdLabel', parent=styles['Normal'],
                                          fontSize=9, textColor=colors.grey, spaceAfter=2)
        prod_code_style  = ParagraphStyle('ProdCode', parent=styles['Normal'],
                                          fontSize=22, textColor=navy, fontName='Helvetica-Bold',
                                          spaceAfter=2, leading=26)
        prod_name_style  = ParagraphStyle('ProdName', parent=styles['Normal'],
                                          fontSize=13, textColor=teal, fontName='Helvetica-Bold',
                                          spaceAfter=6, leading=18)
        prod_desc_style  = ParagraphStyle('ProdDesc', parent=styles['Normal'],
                                          fontSize=10, leading=15, spaceAfter=4)
        prod_meta_style  = ParagraphStyle('ProdMeta', parent=styles['Normal'],
                                          fontSize=9, textColor=colors.HexColor('#555555'),
                                          leading=13, spaceAfter=2)

        story.append(Paragraph('PRODUCT CODE', prod_label_style))
        story.append(Paragraph(pc_code, prod_code_style))
        if pc_name:
            story.append(Paragraph(pc_name, prod_name_style))

        # Date range inputted
        date_range_str = ''
        if pc_date_from and pc_date_to:
            date_range_str = f"{pc_date_from} &nbsp;to&nbsp; {pc_date_to}"
        elif yr_from and yr_to:
            date_range_str = f"{yr_from} – {yr_to}"
        if date_range_str:
            story.append(Paragraph(f"<b>Date Range:</b> {date_range_str}", prod_meta_style))

        meta_parts = []
        if pc_specialty:
            meta_parts.append(f"<b>Medical Specialty:</b> {pc_specialty}")
        if pc_class:
            meta_parts.append(f"<b>Device Class:</b> {pc_class}")
        if pc_reg:
            meta_parts.append(f"<b>Regulation No.:</b> {pc_reg}")
        if meta_parts:
            story.append(Paragraph(' &nbsp;|&nbsp; '.join(meta_parts), prod_meta_style))

        if pc_definition:
            trunc = pc_definition[:500] + ('…' if len(pc_definition) > 500 else '')
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(f"<i>{trunc}</i>", prod_desc_style))

        story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#c7d2e8'),
                                spaceBefore=10, spaceAfter=12))

    story.append(Paragraph(
        "This report presents an analysis of adverse event data from the FDA Manufacturer "
        "and User Facility Device Experience (MAUDE) database. Events are classified using the "
        "International Medical Device Regulators Forum (IMDRF) coding system. The report "
        "covers device problem codes (A-codes), patient problem descriptors, and event outcome "
        "classifications for the selected period.",
        body_style,
    ))
    story.append(Spacer(1, 0.5*cm))

    # Summary metrics box
    top_code = fams[0] if fams else None
    meta_rows = [
        ['Total MAUDE events analysed', f'{total:,}'],
        ['Total manufacturers identified', f'{n_mfrs:,}'],
        ['Analysis period', f'{yr_from} – {yr_to}'],
    ]
    if top_code:
        meta_rows.append(['Top code family', f"{top_code['code']} — {top_code['description']}"])
    meta_tbl = Table(meta_rows, colWidths=[page_w * 0.45, page_w * 0.55])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (0, -1), mid_bg),
        ('FONTNAME',    (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 10),
        ('ALIGN',       (0, 0), (-1, -1), 'LEFT'),
        ('GRID',        (0, 0), (-1, -1), 0.4, colors.HexColor('#b0c4de')),
        ('TOPPADDING',  (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_tbl)
    story.append(PageBreak())

    # ── Section 1 — Top IMDRF Code Families ──────────────────────────────────
    story.append(Paragraph('1. Top IMDRF Code Families', h1_style))
    story.append(HRFlowable(width='100%', thickness=1, color=navy, spaceAfter=8))
    story.append(Paragraph(
        "The table below summarises the top device-problem code families (Level-1 A-codes) "
        "identified in the dataset, ranked by total event count across the analysis period. "
        "Codes A24 and A25 (device investigation outcomes) are excluded as per standard practice.",
        body_style,
    ))

    # Overview table: Code | Description | Grand Total | % of All
    if fams:
        pct_rows = [
            [f['code'],
             Paragraph(f['description'] or '—', ParagraphStyle('td', fontSize=9, leading=12)),
             f"{f.get('family_total', f['grand_total']):,}",
             f"{f.get('family_total', f['grand_total'])/total*100:.1f}%" if total else '—']
            for f in fams
        ]
        col_w = [page_w*0.12, page_w*0.52, page_w*0.18, page_w*0.18]
        story.append(make_table(
            ['Code', 'Description', 'Total Events', '% of All Events'],
            pct_rows, col_w,
        ))
    story.append(Spacer(1, 8))

    # Year-over-year grouped bar chart
    yoy_img = embed_image(chart_images.get('yoy_bar'), width=page_w, height=page_w * 0.45)
    if yoy_img:
        story.append(yoy_img)
        story.append(Paragraph('Figure 1: Year-over-year event counts for top code families.',
                                caption_style))

    # Total comparison bar chart
    tot_img = embed_image(chart_images.get('total_bar'), width=page_w, height=page_w * 0.38)
    if tot_img:
        story.append(tot_img)
        story.append(Paragraph('Figure 2: Total event count comparison across top code families.',
                                caption_style))

    story.append(PageBreak())

    # Per-family subsections
    story.append(Paragraph('1.1 Code Family Details', h2_style))
    for fam in fams:
        code = fam['code']
        desc = fam['description'] or code
        story.append(Paragraph(f"{code} — {desc}", h3_style))

        hierarchy   = fam.get('full_hierarchy', [])
        grand_total = fam['grand_total']

        # ── Year-by-year event summary ────────────────────────────────────────
        yr_header = [''] + [str(y) for y in years]
        yr_row    = ['Events'] + [f"{c:,}" if c else '—' for c in fam['yearly_counts']]
        col_w_yr  = [page_w * 0.14] + [page_w * 0.86 / max(len(years), 1)] * len(years)
        story.append(make_table(yr_header, [yr_row], col_w_yr))
        story.append(Spacer(1, 6))

        # ── Full L1/L2/L3 hierarchy breakdown table ───────────────────────────
        # Format: Code | Level | Description | Count  (total across full period)
        # matching the Excel distribution view
        col_widths_h = [page_w * 0.13, page_w * 0.12, page_w * 0.63, page_w * 0.12]

        hdr_s  = ParagraphStyle('HHdr',  parent=styles['Normal'],
                                fontSize=9, alignment=1, textColor=colors.white,
                                fontName='Helvetica-Bold')
        l1_s   = ParagraphStyle('HL1',   parent=styles['Normal'],
                                fontSize=9, fontName='Helvetica-Bold',
                                textColor=navy, leading=12)
        l2_s   = ParagraphStyle('HL2',   parent=styles['Normal'],
                                fontSize=9, leading=12, textColor=teal)
        l3_s   = ParagraphStyle('HL3',   parent=styles['Normal'],
                                fontSize=8, leading=11,
                                textColor=colors.HexColor('#333333'))
        desc_s = ParagraphStyle('HDesc', parent=styles['Normal'],
                                fontSize=9, leading=11)
        gt_s   = ParagraphStyle('HGT',   parent=styles['Normal'],
                                fontSize=9, fontName='Helvetica-Bold',
                                textColor=colors.white)

        h_header = [
            Paragraph('Code',        hdr_s),
            Paragraph('Level',       hdr_s),
            Paragraph('Description', hdr_s),
            Paragraph('Count',       hdr_s),
        ]

        level_bg = {
            'Level 1': colors.HexColor('#dce8f8'),
            'Level 2': colors.HexColor('#f0f4ff'),
            'Level 3': colors.white,
        }

        h_data_rows   = []  # cell data
        h_row_bgs     = []  # (tbl_row_index, bg_color) — index includes header at 0

        for row in hierarchy:
            lvl = row['level']
            if   lvl == 'Level 1': rs = l1_s
            elif lvl == 'Level 2': rs = l2_s
            else:                   rs = l3_s
            cnt = row['total']
            h_data_rows.append([
                Paragraph(row['code'],                      rs),
                Paragraph(lvl,                              rs),
                Paragraph(row['description'] or '—',       desc_s),
                Paragraph(f"{cnt:,}" if cnt else '—',      rs),
            ])
            # tbl row index = 1-based data index + 1 header row = len(h_data_rows) after append
            h_row_bgs.append((len(h_data_rows), level_bg.get(lvl, colors.white)))

        # Grand Total row — appended last; its tbl index = len(h_data_rows) + 1 (header)
        h_data_rows.append([
            Paragraph('Grand Total', gt_s),
            Paragraph('',            gt_s),
            Paragraph('',            gt_s),
            Paragraph(f"{grand_total:,}", gt_s),
        ])
        gt_tbl_idx = len(h_data_rows)  # = n_data_rows (after append); +header(0) = this index

        tbl_data = [h_header] + h_data_rows
        hier_tbl = Table(tbl_data, colWidths=col_widths_h, repeatRows=1)

        ts_cmds = [
            ('BACKGROUND',    (0, 0),  (-1, 0),              navy),
            ('ALIGN',         (0, 0),  (-1, -1),             'CENTER'),
            ('ALIGN',         (0, 1),  (2, -2),              'LEFT'),
            ('GRID',          (0, 0),  (-1, -1),             0.3, colors.HexColor('#b0c4de')),
            ('TOPPADDING',    (0, 0),  (-1, -1),             3),
            ('BOTTOMPADDING', (0, 0),  (-1, -1),             3),
            ('LEFTPADDING',   (0, 0),  (-1, -1),             4),
            # Grand Total row — always last row in table
            ('BACKGROUND',    (0, gt_tbl_idx), (-1, gt_tbl_idx), navy),
            ('TEXTCOLOR',     (0, gt_tbl_idx), (-1, gt_tbl_idx), colors.white),
            ('FONTNAME',      (0, gt_tbl_idx), (-1, gt_tbl_idx), 'Helvetica-Bold'),
        ]
        # Per-row backgrounds (applied after GT so GT takes priority via later append)
        for (ri, bg) in h_row_bgs:
            if ri != gt_tbl_idx:   # never overwrite GT row
                ts_cmds.append(('BACKGROUND', (0, ri), (-1, ri), bg))

        hier_tbl.setStyle(TableStyle(ts_cmds))
        story.append(hier_tbl)
        story.append(Spacer(1, 8))

        # Top manufacturers table
        if fam['top_manufacturers']:
            story.append(Paragraph('Top Manufacturers:', body_style))
            _mfr_denom = fam.get('family_total') or fam['grand_total']
            mfr_rows = [[m['name'], f"{m['count']:,}",
                         f"{m['count']/_mfr_denom*100:.1f}%" if _mfr_denom else '—']
                        for m in fam['top_manufacturers']]
            story.append(make_table(
                ['Manufacturer', 'Events', '% of Code Total'],
                mfr_rows,
                [page_w*0.60, page_w*0.20, page_w*0.20],
            ))

        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ── Section 2 — Patient Problems & Event Outcomes ─────────────────────────
    story.append(Paragraph('2. Patient Problems & Event Outcomes', h1_style))
    story.append(HRFlowable(width='100%', thickness=1, color=navy, spaceAfter=8))

    # 2a — Patient problems (E-code descriptors)
    story.append(Paragraph('2.1 Patient-Reported Problems', h2_style))
    story.append(Paragraph(
        "Patient problem descriptors, as reported in MAUDE submissions, are summarised below. "
        "These correspond to IMDRF Annex E patient problem codes and represent the most frequently "
        "cited adverse outcomes experienced by patients.",
        body_style,
    ))

    if pp:
        top_pp = sorted(pp.items(), key=lambda x: -x[1])[:15]
        pp_total = sum(pp.values())
        pp_rows = [
            [Paragraph(prob[:80], ParagraphStyle('pprow', fontSize=9, leading=12)),
             f"{cnt:,}",
             f"{cnt/pp_total*100:.1f}%" if pp_total else '—']
            for prob, cnt in top_pp
        ]
        story.append(make_table(
            ['Patient Problem', 'Reports', '% of All'],
            pp_rows,
            [page_w*0.65, page_w*0.17, page_w*0.18],
            alt_color=accent,
        ))
    else:
        story.append(Paragraph('No patient problem data available in this dataset.', note_style))

    # Patient problems chart
    pp_img = embed_image(chart_images.get('patient_problems_bar'), width=page_w, height=page_w * 0.48)
    if pp_img:
        story.append(pp_img)
        story.append(Paragraph('Figure 3: Top 10 patient-reported problems by frequency.',
                                caption_style))

    # 2b — Event outcome types (Annex F proxy)
    story.append(Paragraph('2.2 Event Outcome Classification', h2_style))
    story.append(Paragraph(
        "The table below shows event outcome classifications as recorded in the MAUDE reports. "
        "These categories correspond broadly to IMDRF Annex F patient outcome codes and include "
        "injury, malfunction, death, and other classifications.",
        body_style,
    ))

    if et:
        et_total = sum(et.values())
        et_rows = [
            [etype, f"{cnt:,}", f"{cnt/et_total*100:.1f}%" if et_total else '—']
            for etype, cnt in sorted(et.items(), key=lambda x: -x[1])
        ]
        story.append(make_table(
            ['Event Type', 'Count', '% of All Events'],
            et_rows,
            [page_w*0.55, page_w*0.22, page_w*0.23],
            alt_color=accent,
        ))
    else:
        story.append(Paragraph('No event type classification data available in this dataset.',
                                note_style))

    story.append(PageBreak())

    # ── Section 3 — Conclusion ────────────────────────────────────────────────
    story.append(Paragraph('3. Conclusion', h1_style))
    story.append(HRFlowable(width='100%', thickness=1, color=navy, spaceAfter=8))

    top_code_str = (
        f"<b>{fams[0]['code']}</b> ({fams[0]['description']})"
        if fams else "the analysed codes"
    )
    yr_range_str = f"{yr_from}" if yr_from == yr_to else f"{yr_from}–{yr_to}"

    conclusion = (
        f"This report analysed <b>{total:,}</b> IMDRF-coded adverse event records from the FDA "
        f"MAUDE database for the period <b>{yr_range_str}</b>, involving "
        f"<b>{n_mfrs:,}</b> distinct manufacturers. The predominant device problem code family "
        f"was {top_code_str}, which accounted for the highest volume of events across the "
        f"analysis period."
    )
    story.append(Paragraph(conclusion, body_style))

    if fams and len(years) > 1:
        # Year-over-year commentary — report endpoint change and flag peak
        # years when the trend isn't monotonic (otherwise the headline
        # % hides the real shape of the data).
        trend_comments = []
        for fam in fams[:3]:
            yr_vals = fam['yearly_counts']
            if len(yr_vals) < 2 or yr_vals[0] <= 0:
                continue
            change = (yr_vals[-1] - yr_vals[0]) / yr_vals[0] * 100
            direction = 'increased' if change > 0 else 'decreased'
            peak_idx = max(range(len(yr_vals)), key=lambda i: yr_vals[i])
            peak_val = yr_vals[peak_idx]
            peak_year = years[peak_idx]
            peak_note = ''
            # Non-monotonic: peak is at an interior year AND materially larger
            # than both endpoints (>20% above the higher endpoint).
            endpoints_max = max(yr_vals[0], yr_vals[-1])
            if (peak_idx not in (0, len(yr_vals) - 1)
                    and endpoints_max > 0
                    and peak_val > endpoints_max * 1.2):
                peak_note = (
                    f" Activity peaked in <b>{peak_year}</b> at "
                    f"<b>{peak_val:,}</b> events before declining."
                )
            trend_comments.append(
                f"Code <b>{fam['code']}</b> ({fam['description'] or fam['code']}) "
                f"{direction} by <b>{abs(change):.0f}%</b> from {yr_from} to {yr_to} "
                f"({yr_vals[0]:,} → {yr_vals[-1]:,} events).{peak_note}"
            )
        if trend_comments:
            story.append(Paragraph('Year-over-year trend observations:', body_style))
            for comment in trend_comments:
                story.append(Paragraph(f"• {comment}", body_style))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "This report is generated from publicly available FDA MAUDE data and is intended for "
        "signal detection and trend monitoring purposes only. It does not constitute a formal "
        "regulatory submission or safety conclusion.",
        note_style,
    ))

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey, spaceAfter=4))
    story.append(Paragraph(
        f"Generated by FDA MAUDE Analysis Platform · {dt.now().strftime('%d %B %Y, %H:%M UTC')}",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                       textColor=colors.grey, alignment=1),
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
