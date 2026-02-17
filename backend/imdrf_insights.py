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
    - Level-2: 5 characters (e.g., "A0501"), includes truncated Level-3 codes
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


def aggregate_by_grain(df, date_col, grain='W'):
    """
    Aggregate counts by date grain.

    Args:
        df: DataFrame with parsed dates
        date_col: Name of the parsed date column
        grain: 'D' (daily), 'W' (weekly), 'M' (monthly)

    Returns:
        Series with date index and counts
    """
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
    date_patterns_exact = [
        "event date", "event_date", "eventdate",
        "date received", "date_received", "datereceived",
        "report date", "report_date", "reportdate",
        "date", "mdr_date", "received_date"
    ]

    date_patterns_contains = [
        "date", "received", "event"
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


def analyze_imdrf_insights(df, selected_prefix, selected_manufacturers, grain='W', threshold_k=2.0, level=1):
    """
    Perform IMDRF prefix insights analysis.

    Args:
        df: DataFrame with exploded IMDRF prefixes and parsed dates
        selected_prefix: IMDRF prefix to analyze (e.g., "A05" for Level-1, "A0501" for Level-2)
        selected_manufacturers: List of manufacturer names to compare
        grain: Date aggregation grain ('D', 'W', 'M')
        threshold_k: Standard deviation multiplier for thresholds
        level: 1, 2, or 3 - IMDRF analysis level (affects universal mean calculation)

    Returns:
        dict with analysis results including:
        - universal_mean: Mean across all IMDRF events at the specified level
        - prefix_mean: Mean for selected prefix
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

    # A) Universal mean baseline (all IMDRF-coded events, all prefixes)
    universal_counts = aggregate_by_grain(df_with_dates, 'parsed_date', grain)
    universal_mean = float(universal_counts.mean()) if len(universal_counts) > 0 else 0.0

    # Filter to selected prefix
    df_prefix = df_with_dates[df_with_dates['imdrf_prefix'] == selected_prefix].copy()

    if df_prefix.empty:
        raise ValueError(f"No data found for prefix '{selected_prefix}'")

    # B) Prefix-specific mean baseline (selected prefix, all manufacturers)
    prefix_counts = aggregate_by_grain(df_prefix, 'parsed_date', grain)
    prefix_mean = float(prefix_counts.mean()) if len(prefix_counts) > 0 else 0.0
    prefix_std = float(prefix_counts.std()) if len(prefix_counts) > 0 else 0.0

    # Calculate thresholds
    upper_threshold = prefix_mean + threshold_k * prefix_std
    lower_threshold = max(0.0, prefix_mean - threshold_k * prefix_std)

    # Filter to selected manufacturers
    df_selected = df_prefix[df_prefix[mfr_col].isin(selected_manufacturers)].copy()

    if df_selected.empty:
        raise ValueError(f"No data found for selected manufacturers with prefix '{selected_prefix}'")

    # Create time-series for each manufacturer
    manufacturer_series = {}
    all_dates = set()

    for mfr in selected_manufacturers:
        df_mfr = df_selected[df_selected[mfr_col] == mfr]
        mfr_counts = aggregate_by_grain(df_mfr, 'parsed_date', grain)
        manufacturer_series[mfr] = mfr_counts
        if len(mfr_counts) > 0:
            all_dates.update(mfr_counts.index)

    # Create a complete date range
    if all_dates:
        date_range = pd.date_range(
            start=min(all_dates),
            end=max(all_dates),
            freq=grain
        )

        # Reindex each manufacturer series to fill gaps with 0
        for mfr in manufacturer_series:
            manufacturer_series[mfr] = manufacturer_series[mfr].reindex(date_range, fill_value=0)
    else:
        date_range = pd.DatetimeIndex([])

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
        "universal_mean": round(universal_mean, 2),
        "prefix_mean": round(prefix_mean, 2),
        "prefix_std": round(prefix_std, 2),
        "upper_threshold": round(upper_threshold, 2),
        "lower_threshold": round(lower_threshold, 2),
        "manufacturer_series": manufacturer_series,
        "date_range": date_range,
        "statistics": statistics,
        "grain": grain,
        "selected_prefix": selected_prefix,
        "level": level,
        "level_label": level_label
    }


def prepare_data_for_insights(file_path, level=1):
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
        df = pd.read_excel(file_path, dtype=str)
    elif file_ext == '.csv':
        df = pd.read_csv(file_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please upload CSV, XLS, or XLSX file.")

    # Clean up string columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "NaN": "", "None": ""})

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

    if df_with_dates.empty:
        sample_values = df[date_col].head(5).tolist()
        raise ValueError(
            f"No parsable dates found in '{date_col}' column. "
            f"Sample values: {sample_values}. "
            f"Supported formats: DD-MM-YYYY, YYYYMMDD, YYYY-MM-DD, MM/DD/YYYY"
        )

    # Get unique prefixes and manufacturers
    prefix_counts_series = df_with_dates['imdrf_prefix'].value_counts()
    prefix_counts = prefix_counts_series.to_dict()
    all_prefixes = sorted(prefix_counts.keys())
    all_manufacturers = sorted([m for m in df_with_dates[mfr_col].unique() if m and str(m).strip()])

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
        "level_label": level_label
    }


def _load_cleaned_dataframe(file_path: str) -> pd.DataFrame:
    """Load a cleaned MAUDE file into a DataFrame with string values."""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, dtype=str)
    elif file_ext == '.csv':
        df = pd.read_csv(file_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please upload CSV, XLS, or XLSX file.")

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "NaN": "", "None": ""})

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
        counts = exploded.value_counts().to_dict()
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
