"""
IMDRF Annex Code Validator Module

This module parses the "Annexes A-G consolidated.xlsx" file to extract
valid IMDRF codes at Level-1 (3 chars), Level-2 (5 chars), and Level-3 (7 chars).

The Annex file contains sheets A through G with IMDRF code definitions.
"""

import os
import re
from typing import Dict, Set, Optional, Tuple
from functools import lru_cache


# Default path for the Annex file (can be overridden via environment variable)
DEFAULT_ANNEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Annexes A-G consolidated.xlsx')
ANNEX_FILE_PATH = os.environ.get('IMDRF_ANNEX_FILE', DEFAULT_ANNEX_PATH)


class IMDRFAnnexValidator:
    """
    Validates IMDRF codes against the authoritative Annex file.

    Supports three validation levels:
    - Level-1: First 3 alphanumeric characters (e.g., "A01", "E24")
    - Level-2: First 5 alphanumeric characters (e.g., "A0101", "E2401")
    - Level-3: First 7 alphanumeric characters (e.g., "A010101", "E240101")
    """

    def __init__(self, annex_file_path: str = None):
        """
        Initialize the validator.

        Args:
            annex_file_path: Path to the Annex Excel file. Uses default if not specified.
        """
        self.annex_file_path = annex_file_path or ANNEX_FILE_PATH
        self._valid_level1: Set[str] = set()
        self._valid_level2: Set[str] = set()
        self._valid_level3: Set[str] = set()
        self._loaded = False
        self._load_error: Optional[str] = None

    def _extract_code_from_cell(self, cell_value) -> Optional[str]:
        """
        Extract alphanumeric IMDRF code from a cell value.

        Args:
            cell_value: Cell value (may be string, number, or None)

        Returns:
            Cleaned alphanumeric code string or None
        """
        if cell_value is None:
            return None

        # Convert to string and strip
        s = str(cell_value).strip()

        if not s or s.lower() in ('nan', 'none', 'nat', ''):
            return None

        # Extract only alphanumeric characters
        alphanumeric = re.sub(r'[^A-Za-z0-9]', '', s)

        if len(alphanumeric) < 3:
            return None

        return alphanumeric.upper()

    def load_annex_file(self) -> bool:
        """
        Load and parse the Annex Excel file.

        Reads all sheets (A through G) and extracts valid IMDRF codes
        at each level (1, 2, 3).

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        if not os.path.exists(self.annex_file_path):
            self._load_error = f"Annex file not found: {self.annex_file_path}"
            return False

        try:
            import pandas as pd

            # Read all sheets from the Excel file
            xlsx = pd.ExcelFile(self.annex_file_path)

            for sheet_name in xlsx.sheet_names:
                try:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name, dtype=str)

                    # Look for code columns - typically the first column or columns named "Code", "IMDRF Code", etc.
                    code_columns = []
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'code' in col_lower or col == df.columns[0]:
                            code_columns.append(col)

                    if not code_columns:
                        code_columns = [df.columns[0]] if len(df.columns) > 0 else []

                    # Extract codes from each potential code column
                    for col in code_columns:
                        for value in df[col].dropna():
                            code = self._extract_code_from_cell(value)
                            if code:
                                # Add to appropriate level sets based on code length
                                if len(code) >= 3:
                                    self._valid_level1.add(code[:3])
                                if len(code) >= 5:
                                    self._valid_level2.add(code[:5])
                                if len(code) >= 7:
                                    self._valid_level3.add(code[:7])

                except Exception as e:
                    # Continue processing other sheets even if one fails
                    continue

            self._loaded = True
            return True

        except ImportError:
            self._load_error = "pandas or openpyxl not installed for Excel file reading"
            return False
        except Exception as e:
            self._load_error = f"Error loading Annex file: {str(e)}"
            return False

    def get_valid_codes(self, level: int) -> Set[str]:
        """
        Get the set of valid codes for a specific level.

        Args:
            level: 1, 2, or 3

        Returns:
            Set of valid codes for that level
        """
        if not self._loaded:
            self.load_annex_file()

        if level == 1:
            return self._valid_level1.copy()
        elif level == 2:
            return self._valid_level2.copy()
        elif level == 3:
            return self._valid_level3.copy()
        else:
            raise ValueError(f"Invalid level: {level}. Must be 1, 2, or 3.")

    def is_valid_code(self, code: str, level: int) -> bool:
        """
        Check if a code is valid at a specific level.

        Args:
            code: The IMDRF code to validate
            level: 1, 2, or 3

        Returns:
            True if valid, False otherwise
        """
        if not self._loaded:
            self.load_annex_file()

        code = code.upper()

        if level == 1:
            return code[:3] in self._valid_level1 if len(code) >= 3 else False
        elif level == 2:
            return code[:5] in self._valid_level2 if len(code) >= 5 else False
        elif level == 3:
            return code[:7] in self._valid_level3 if len(code) >= 7 else False
        else:
            raise ValueError(f"Invalid level: {level}. Must be 1, 2, or 3.")

    def get_load_error(self) -> Optional[str]:
        """Get the last load error message, if any."""
        return self._load_error

    def is_loaded(self) -> bool:
        """Check if the Annex file has been loaded."""
        return self._loaded

    def get_stats(self) -> Dict:
        """
        Get statistics about the loaded codes.

        Returns:
            Dictionary with counts for each level
        """
        if not self._loaded:
            self.load_annex_file()

        return {
            'level1_count': len(self._valid_level1),
            'level2_count': len(self._valid_level2),
            'level3_count': len(self._valid_level3),
            'loaded': self._loaded,
            'error': self._load_error
        }


# Global singleton instance
_validator_instance: Optional[IMDRFAnnexValidator] = None


def get_validator(annex_file_path: str = None) -> IMDRFAnnexValidator:
    """
    Get the global validator instance.

    Args:
        annex_file_path: Optional path to override default Annex file location

    Returns:
        IMDRFAnnexValidator instance
    """
    global _validator_instance

    if _validator_instance is None or (annex_file_path and annex_file_path != _validator_instance.annex_file_path):
        _validator_instance = IMDRFAnnexValidator(annex_file_path)

    return _validator_instance


def extract_imdrf_codes_by_level(imdrf_code_str: str, level: int, validator: IMDRFAnnexValidator = None) -> list:
    """
    Extract IMDRF codes from a string at the specified level.

    This is the main extraction function that handles:
    - Pipe-separated multi-code cells
    - Level-specific extraction (3, 5, or 7 characters)
    - Validation against Annex file (for Level-3, only exact matches)
    - For Level-2: includes Level-3 codes truncated to 5 chars

    Args:
        imdrf_code_str: String containing IMDRF codes (may be pipe-separated)
        level: 1, 2, or 3
        validator: Optional validator instance. If None, uses global validator.

    Returns:
        List of extracted codes at the specified level
    """
    import pandas as pd

    if pd.isna(imdrf_code_str):
        return []

    s = str(imdrf_code_str).strip()
    if not s or s.lower() in ('nan', 'nat', 'none', ''):
        return []

    # Get validator if not provided
    if validator is None:
        validator = get_validator()

    # Determine required length based on level
    level_lengths = {1: 3, 2: 5, 3: 7}
    required_length = level_lengths.get(level, 3)

    # Split on pipe
    tokens = s.split('|')
    codes = []

    for token in tokens:
        token = token.strip()

        # Extract only alphanumeric characters
        alphanumeric = re.sub(r'[^A-Za-z0-9]', '', token)

        if len(alphanumeric) < required_length:
            continue

        code = alphanumeric[:required_length].upper()

        # Validation logic based on level
        if level == 3:
            # Level-3: Only exact matches from valid_level3 set
            if validator.is_loaded() and validator.get_valid_codes(3):
                if code in validator.get_valid_codes(3):
                    codes.append(code)
            else:
                # If no Annex file, accept any 7-char code
                codes.append(code)
        elif level == 2:
            # Level-2: Accept 5-char codes, including truncated Level-3 codes
            codes.append(code)
        else:
            # Level-1: Accept 3-char codes (existing behavior)
            codes.append(code)

    return codes


def get_annex_status() -> Dict:
    """
    Get the status of the Annex file loading.

    Returns:
        Dictionary with status information
    """
    validator = get_validator()

    return {
        'file_path': validator.annex_file_path,
        'file_exists': os.path.exists(validator.annex_file_path),
        'loaded': validator.is_loaded(),
        'error': validator.get_load_error(),
        'stats': validator.get_stats() if validator.is_loaded() else None
    }
