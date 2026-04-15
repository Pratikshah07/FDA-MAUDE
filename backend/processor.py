"""
Main processing pipeline for MAUDE data cleaning and enrichment.
Implements all phases: ingestion, cleaning, standardization, and enrichment.
Deterministic and audit-defensible.

Scalability Features:
- Chunked file reading for large CSV files (>50MB)
- Memory optimization using category dtype for low-cardinality columns
- Batch IMDRF mapping (processes unique problems once, not per-row)
- Deterministic-only mode for files with 1000+ rows (fast processing)
"""
import os
import pandas as pd
import re
from datetime import datetime
from typing import Optional, Dict, List, Callable
from dateutil import parser as date_parser

from config import (
    COLUMNS_TO_DELETE,
    COLUMNS_TO_MODIFY,
    COLUMNS_TO_ADD,
    LEGAL_SUFFIXES,
    DATE_FORMAT
)
from backend.groq_client import GroqClient
from backend.column_identifier import ColumnIdentifier
from backend.imdrf_mapper import IMDRFMapper
from backend.manufacturer_normalizer import ManufacturerNormalizer


class MAUDEProcessor:
    """Main processor for MAUDE data pipeline."""
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        self.groq_client = groq_client or GroqClient()
        # Initialize specialized components
        self.column_identifier = ColumnIdentifier(self.groq_client)
        self.imdrf_mapper = IMDRFMapper(self.groq_client)
        self.manufacturer_normalizer = ManufacturerNormalizer(self.groq_client)
        
        # Column mappings (resolved by ColumnIdentifier)
        self.column_map = {}

        # Audit tracking (populated during process_file phases)
        self._audit_cols_removed: List[str] = []
        self._audit_rows_removed_by_reason: List[Dict] = []   # [{reason, count}]
        self._audit_manufacturer_list: List[str] = []
        self._audit_imdrf_stats: Dict = {}                    # {non_empty, mapped, unmapped}
        # Optional callback for phase-level progress updates: fn(phase_label: str)
        self.phase_callback: Optional[Callable[[str], None]] = None

    def _emit_phase(self, label: str):
        if self.phase_callback:
            try:
                self.phase_callback(label)
            except Exception:
                pass
    
    def load_imdrf_structure(self, file_path: str):
        """
        Load IMDRF Annexure A-G structure from file using deterministic parsing.
        """
        try:
            self.imdrf_mapper.load_annex(file_path)
        except Exception as e:
            print(f"ERROR: Could not load IMDRF structure: {e}")
            import traceback
            traceback.print_exc()
    
    
    def process_file(self, input_path: str, output_path: str) -> Dict[str, any]:
        """
        Process MAUDE file through all phases.
        
        Returns:
            Dictionary with processing statistics and validation results
        """
        # Phase 0: Column Identification (AI-assisted)
        self._emit_phase('Reading raw data…')
        df = self._ingest_file(input_path)
        
        if df.empty:
            raise ValueError("Input file is empty or contains no data")
        
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # Identify columns using deterministic + Groq fallback
        print("\n=== COLUMN IDENTIFICATION PHASE ===")
        self.column_map = self.column_identifier.identify_columns(list(df.columns))
        print(f"Column mapping: {self.column_map}")
        
        # Phase 1: Normalize missing tokens
        self._emit_phase('Normalizing missing values…')
        df = self._normalize_missing_tokens(df)

        # Phase 2: Data Cleaning (column removal only)
        self._emit_phase('Removing unused columns…')
        df = self._clean_data(df)

        # Phase 3: Date Standardization (converts to "" on failure)
        self._emit_phase('Standardizing dates…')
        df = self._standardize_dates(df)

        # Phase 3.5: Row Removal (AFTER date standardization)
        df = self._remove_blank_date_rows(df)

        # Phase 4: Manufacturer Normalization
        self._emit_phase('Normalizing manufacturers…')
        df = self._normalize_manufacturers(df)
        
        # Phase 5: Extract Keywords - DISABLED per user request
        # df = self._extract_keywords(df)
        
        # Phase 6: IMDRF Mapping (Deterministic primary + Groq fallback)
        self._emit_phase('Mapping IMDRF codes for device problems…')
        print(f"\n=== IMDRF MAPPING PHASE ===")
        total_codes = len(self.imdrf_mapper.level1_map) + len(self.imdrf_mapper.level2_map) + len(self.imdrf_mapper.level3_map)
        if total_codes > 0:
            print(f"IMDRF lookup maps loaded: Level-1={len(self.imdrf_mapper.level1_map)}, Level-2={len(self.imdrf_mapper.level2_map)}, Level-3={len(self.imdrf_mapper.level3_map)}")
        else:
            print("WARNING: IMDRF lookup maps are empty. IMDRF codes will be blank.")
        df = self._map_imdrf_codes(df)

        # Phase 6.5: Patient Problem normalization (explode semicolon-separated values into rows)
        self._emit_phase('Mapping IMDRF codes for patient problems…')
        df = self._explode_patient_problem_rows(df)

        # Phase 7: Final Validation
        self._emit_phase('Validating output…')
        validation_results = self._validate_output(df, original_cols)

        # Phase 7.5: Sanitize for Excel output (remove illegal characters)
        self._emit_phase('Sanitizing cells for Excel…')
        df = self._sanitize_for_excel(df)

        # Save output (prefer faster xlsxwriter if available)
        self._emit_phase(f'Writing Excel file ({len(df):,} rows)…')
        try:
            import xlsxwriter  # noqa: F401
            df.to_excel(output_path, index=False, engine='xlsxwriter')
            print(f"Excel written via xlsxwriter: {output_path}")
        except ImportError:
            print("xlsxwriter not installed — falling back to openpyxl (slower). pip install xlsxwriter recommended.")
            df.to_excel(output_path, index=False, engine='openpyxl')
        except Exception as e:
            print(f"xlsxwriter failed ({e}); retrying with openpyxl…")
            df.to_excel(output_path, index=False, engine='openpyxl')
        
        return {
            'original_rows': original_rows,
            'final_rows': len(df),
            'original_cols': original_cols,
            'final_cols': len(df.columns),
            'rows_removed': original_rows - len(df),
            'validation': validation_results,
            # Audit fields
            'cols_removed': self._audit_cols_removed,
            'rows_removed_by_reason': self._audit_rows_removed_by_reason,
            'manufacturer_list': self._audit_manufacturer_list,
            'imdrf_stats': self._audit_imdrf_stats,
        }

    def _sanitize_for_excel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove illegal characters that Excel worksheets cannot accept.
        This prevents openpyxl IllegalCharacterError during export.
        """
        ILLEGAL_CHARACTERS_RE = None
        for _mod in ('openpyxl.cell.cell', 'openpyxl.utils.cell'):
            try:
                ILLEGAL_CHARACTERS_RE = __import__(_mod, fromlist=['ILLEGAL_CHARACTERS_RE']).ILLEGAL_CHARACTERS_RE
                break
            except Exception:
                continue
        if ILLEGAL_CHARACTERS_RE is None:
            # Compile our own — same ranges Excel rejects
            ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

        if ILLEGAL_CHARACTERS_RE is not None:
            for col in df.columns:
                dt = str(df[col].dtype)
                if dt not in ('object', 'string', 'category'):
                    continue
                if dt == 'category':
                    df[col] = df[col].astype(str)
                # Fast pre-check: skip columns with no illegal chars at all
                try:
                    if not df[col].str.contains(ILLEGAL_CHARACTERS_RE, regex=True, na=False).any():
                        continue
                except Exception:
                    pass
                df[col] = df[col].str.replace(ILLEGAL_CHARACTERS_RE, '', regex=True)
            return df

        def _clean_value(value):
            if value is None:
                return value
            text = str(value)
            return ''.join(ch for ch in text if ch in ('\t', '\n', '\r') or ord(ch) >= 32)

        for col in df.columns:
            df[col] = df[col].map(_clean_value)
        return df
    
    def _ingest_file(self, file_path: str) -> pd.DataFrame:
        """
        Phase 1: Safe file ingestion with encoding detection and file type detection.

        Memory optimizations for large files:
        - Uses category dtype for low-cardinality columns
        - Processes in chunks for very large CSV files
        - Garbage collection after loading
        """
        import gc

        try:
            # Get file size for optimization decisions
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"Ingesting file: {file_path} ({file_size_mb:.1f} MB)")

            # Detect file type by reading first few bytes (more reliable than extension)
            is_excel = False
            with open(file_path, 'rb') as f:
                header = f.read(8)
                # Excel file signatures
                if header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):  # OLE2 (xls)
                    is_excel = True
                elif header.startswith(b'PK\x03\x04'):  # ZIP signature (xlsx)
                    is_excel = True
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    # Trust extension if header check is ambiguous
                    is_excel = True

            if is_excel:
                # Read as strings to avoid Excel auto-parsing
                # index_col=None prevents pandas from creating an index column
                try:
                    if file_path.endswith('.xlsx') or header.startswith(b'PK\x03\x04'):
                        df = pd.read_excel(file_path, dtype=str, engine='openpyxl', index_col=None)
                    else:
                        df = pd.read_excel(file_path, dtype=str, engine='xlrd', index_col=None)
                except Exception as e:
                    raise ValueError(f"Failed to read Excel file: {e}")
            else:
                # Try multiple encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252', 'utf-16']
                df = None
                last_error = None

                # For large CSV files (>50MB), use chunked reading
                use_chunked = file_size_mb > 50

                for encoding in encodings:
                    try:
                        if use_chunked:
                            # Read in chunks to reduce peak memory
                            print(f"  Using chunked reading for large file...")
                            chunks = []
                            try:
                                chunk_iter = pd.read_csv(
                                    file_path, dtype=str, encoding=encoding,
                                    chunksize=10000, index_col=False, on_bad_lines='skip'
                                )
                            except TypeError:
                                chunk_iter = pd.read_csv(
                                    file_path, dtype=str, encoding=encoding,
                                    chunksize=10000, index_col=False, error_bad_lines=False
                                )

                            for i, chunk in enumerate(chunk_iter):
                                chunks.append(chunk)
                                if (i + 1) % 10 == 0:
                                    print(f"    Read {(i+1) * 10000} rows...")

                            df = pd.concat(chunks, ignore_index=True)
                            del chunks
                            gc.collect()
                        else:
                            # Standard reading for smaller files
                            try:
                                df = pd.read_csv(file_path, dtype=str, encoding=encoding, low_memory=False, index_col=False, on_bad_lines='skip')
                            except TypeError:
                                df = pd.read_csv(file_path, dtype=str, encoding=encoding, low_memory=False, index_col=False, error_bad_lines=False)
                        break  # Success, exit loop
                    except (UnicodeDecodeError, UnicodeError) as e:
                        last_error = e
                        continue  # Try next encoding
                    except Exception as e:
                        # Other errors (not encoding-related), try next encoding
                        last_error = e
                        continue

                if df is None:
                    raise ValueError(f"Failed to read CSV file with any encoding. Last error: {last_error}")

            # Remove any "Unnamed" columns (usually from index columns)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False, na=False)]

            # Ensure all columns are strings
            for col in df.columns:
                df[col] = df[col].astype(str)

            # Memory optimization: convert low-cardinality columns to category dtype
            if len(df) > 1000:
                for col in df.columns:
                    # Check cardinality
                    nunique = df[col].nunique()
                    if nunique < len(df) * 0.1:  # Less than 10% unique values
                        df[col] = df[col].astype('category')

            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            gc.collect()

            return df

        except Exception as e:
            raise ValueError(f"Failed to ingest file: {e}")
    
    def _normalize_missing_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize literal missing tokens to "" for date columns and device_problem/manufacturer columns.
        """
        # Get column names from mapping
        date_cols = []
        if self.column_map.get("event_date"):
            date_cols.append(self.column_map["event_date"])
        if self.column_map.get("date_received"):
            date_cols.append(self.column_map["date_received"])
        
        device_problem_col = self.column_map.get("device_problem")
        manufacturer_col = self.column_map.get("manufacturer")
        
        # Normalize missing tokens
        missing_tokens = ["nan", "null", "none", "n/a", "na"]
        
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: "" if str(x).strip().lower() in missing_tokens else str(x))
        
        if device_problem_col and device_problem_col in df.columns:
            df[device_problem_col] = df[device_problem_col].apply(lambda x: "" if str(x).strip().lower() in missing_tokens else str(x))
        
        if manufacturer_col and manufacturer_col in df.columns:
            df[manufacturer_col] = df[manufacturer_col].apply(lambda x: "" if str(x).strip().lower() in missing_tokens else str(x))
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 2: Data cleaning - column removal only."""
        # Column removal: Delete specified columns if present (normalized matching, but do NOT rename)
        cols_to_remove = []
        for col in df.columns:
            col_normalized = col.lower().replace('_', ' ').replace('-', ' ').strip()
            for col_to_delete in COLUMNS_TO_DELETE:
                delete_normalized = col_to_delete.lower().replace('_', ' ').replace('-', ' ').strip()
                # For PMA/PMN: check if normalized name startswith "pma/pmn"
                if "pma/pmn" in delete_normalized:
                    if col_normalized.startswith("pma/pmn"):
                        cols_to_remove.append(col)
                        break
                # For others: exact or contained match
                elif (col == col_to_delete or 
                      col_normalized == delete_normalized or
                      delete_normalized in col_normalized):
                    cols_to_remove.append(col)
                    break
        
        self._audit_cols_removed = list(cols_to_remove)

        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)

        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 3: Date standardization to DD-MM-YYYY format (remove time, format DD-MM-YYYY else blank)."""
        event_date_col = self.column_map.get("event_date")
        date_received_col = self.column_map.get("date_received")
        
        if event_date_col and event_date_col in df.columns:
            df[event_date_col] = df[event_date_col].apply(lambda x: self._parse_and_format_date(x))
        
        if date_received_col and date_received_col in df.columns:
            df[date_received_col] = df[date_received_col].apply(lambda x: self._parse_and_format_date(x))
        
        return df
    
    def _remove_blank_date_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where BOTH Event Date == "" AND Date Received == "".
        This runs AFTER:
        - Missing tokens → ""
        - Date parsing → "" on failure
        """
        event_date_col = self.column_map.get("event_date")
        date_received_col = self.column_map.get("date_received")
        
        if event_date_col and date_received_col and event_date_col in df.columns and date_received_col in df.columns:
            # Convert to string and check for empty string (after all processing, empty means blank)
            mask = (
                (df[event_date_col].astype(str).str.strip() == '')
            ) & (
                (df[date_received_col].astype(str).str.strip() == '')
            )
            rows_removed = mask.sum()
            if rows_removed > 0:
                self._audit_rows_removed_by_reason.append({
                    'reason': 'Both Event Date and Date Received are blank',
                    'count': int(rows_removed),
                })
                print(f"Removed {rows_removed} rows where both Event Date and Date Received are blank")
            df = df[~mask]
        
        return df
    
    def _parse_and_format_date(self, date_str: str) -> str:
        """Parse date string and format to DD-MM-YYYY, or return blank if fails."""
        if pd.isna(date_str) or not str(date_str).strip() or str(date_str).strip().lower() == 'nan':
            return ""
        
        date_str = str(date_str).strip()
        
        # Remove time component if present
        if ' ' in date_str:
            date_str = date_str.split()[0]
        if 'T' in date_str:
            date_str = date_str.split('T')[0]
        
        # Try unambiguous formats first (before dateutil, which misinterprets
        # YYYY-MM-DD as day-swapped when dayfirst=True)
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y', '%m-%d-%Y']:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime(DATE_FORMAT)
            except ValueError:
                continue

        try:
            # Last resort: dateutil (handles unusual/partial formats)
            parsed_date = date_parser.parse(date_str, dayfirst=True)
            return parsed_date.strftime(DATE_FORMAT)
        except:
            pass
        
        # If all parsing fails, return blank
        return ""
    
    def _find_manufacturer_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Generalized manufacturer column identification (NO RENAMING).
        Returns the exact existing column name or None.
        """
        def normalize_header(header: str) -> str:
            """Normalize header for matching only."""
            return ' '.join(str(header).strip().lower().split())
        
        # Normalize all headers
        normalized_headers = {normalize_header(h): h for h in df.columns}
        
        # Priority 1: Exact normalized match
        exact_matches = {"manufacturer", "manufacturer name", "mfr", "mfr name"}
        for exact in exact_matches:
            if exact in normalized_headers:
                return normalized_headers[exact]
        
        # Priority 2: Contains "manufacturer"
        candidates = []
        for norm, orig in normalized_headers.items():
            if "manufacturer" in norm:
                candidates.append((norm, orig))
        
        if candidates:
            # Choose shortest normalized header (deterministic tie-break)
            candidates.sort(key=lambda x: len(x[0]))
            return candidates[0][1]
        
        # Priority 3: Contains "mfr"
        candidates = []
        for norm, orig in normalized_headers.items():
            if "mfr" in norm:
                candidates.append((norm, orig))
        
        if candidates:
            # Choose shortest normalized header (deterministic tie-break)
            candidates.sort(key=lambda x: len(x[0]))
            return candidates[0][1]
        
        return None

    def _find_patient_problem_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify the Patient Problem column (NO RENAMING).
        Returns the exact existing column name or None.
        """
        def normalize_header(header: str) -> str:
            return ' '.join(str(header).strip().lower().split())

        normalized_headers = {normalize_header(h): h for h in df.columns}

        exact_matches = {
            "patient problem",
            "patient problems",
            "patient problem text",
            "patient_problem",
            "patient_problems",
            "patient_problem_text"
        }
        for exact in exact_matches:
            if exact in normalized_headers:
                return normalized_headers[exact]

        candidates = []
        for norm, orig in normalized_headers.items():
            if "patient" in norm and "problem" in norm:
                candidates.append((norm, orig))

        if candidates:
            candidates.sort(key=lambda x: len(x[0]))
            return candidates[0][1]

        return None

    def _explode_patient_problem_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If Patient Problem column exists, explode semicolon-separated values into
        multiple rows while keeping all other columns consistent.
        """
        patient_problem_col = self._find_patient_problem_column(df)

        if not patient_problem_col:
            return df

        print(f"\n=== PATIENT PROBLEM NORMALIZATION PHASE ===")
        print(f"Found Patient Problem column: '{patient_problem_col}'")

        # Ensure IMDRF Patient Code column exists immediately after Patient Problem
        if 'IMDRF Patient Code' not in df.columns:
            cols = list(df.columns)
            insert_at = cols.index(patient_problem_col) + 1
            cols.insert(insert_at, 'IMDRF Patient Code')
            df = df.reindex(columns=cols)
            df['IMDRF Patient Code'] = ""

        # Pre-compute patient-problem -> E-code map over unique parts to avoid
        # per-row lookups and per-row dict materialization (OOM-safe).
        raw_series = df[patient_problem_col].fillna("").astype(str)
        parts_series = raw_series.apply(
            lambda s: [p.strip() for p in s.split(";") if p.strip()] or [s.strip()]
        )

        unique_parts = set()
        for parts in parts_series:
            for p in parts:
                if p:
                    unique_parts.add(p)
        patient_map = {p: self.imdrf_mapper.map_patient_problem_part(p) for p in unique_parts}

        work = df.copy()
        work[patient_problem_col] = parts_series
        exploded = work.explode(patient_problem_col, ignore_index=True)
        exploded[patient_problem_col] = exploded[patient_problem_col].fillna("").astype(str)
        exploded['IMDRF Patient Code'] = exploded[patient_problem_col].map(
            lambda p: patient_map.get(p, "") if p else ""
        )

        result_df = exploded[list(df.columns)]

        # Audit
        mapped_mask = result_df['IMDRF Patient Code'].astype(str).str.strip() != ''
        non_empty_mask = result_df[patient_problem_col].astype(str).str.strip() != ''
        print(f"Patient problem mapping: {int(non_empty_mask.sum())} non-empty, {int(mapped_mask.sum())} codes mapped")

        return result_df
    
    def _clean_manufacturer(self, val: str) -> str:
        """
        Deterministic suffix cleanup function.
        """
        # Handle missing tokens case-insensitively
        if val is None:
            return ""
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ['nan', 'none', 'null', 'n/a', 'na']:
            return ""
        
        # 1) lowercase
        name = val_str.lower()
        
        # 2) strip whitespace
        name = name.strip()
        
        # 3) collapse multiple spaces
        name = re.sub(r"\s+", " ", name)
        
        if not name:
            return ""
        
        # 4) remove trailing punctuation repeatedly BEFORE suffix logic
        # Remove trailing characters: . , ; : ( ) [ ]
        changed = True
        while changed:
            changed = False
            original = name
            name = re.sub(r"[.,;:()\[\]]+$", "", name).strip()
            if name != original:
                changed = True
        
        if not name:
            return ""
        
        # 5) split into tokens by spaces
        tokens = name.split()
        
        if not tokens:
            return ""
        
        # 6) iteratively remove legal suffix tokens from the end
        # Suffix set: ltd, limited, llp, inc, corp, company, co, gmbh, ag, sa, sarl, bv, plc, pvt, llc
        legal_suffixes = {"ltd", "limited", "llp", "inc", "corp", "company", "co", "gmbh", "ag", "sa", "sarl", "bv", "plc", "pvt", "llc"}
        
        # Remove suffix tokens from the end until last token is not a suffix
        changed = True
        while changed and tokens:
            changed = False
            # Get last token and strip any remaining punctuation
            last_token = tokens[-1].strip()
            last_token_clean = re.sub(r"[.,;:()\[\]]+$", "", last_token)
            
            # Check if last token (after punctuation removal) is a suffix
            if last_token_clean in legal_suffixes:
                tokens.pop()
                changed = True
        
        # 7) Clean punctuation from all tokens before rejoining
        cleaned_tokens = []
        for token in tokens:
            # Remove trailing punctuation from each token
            cleaned = re.sub(r"[.,;:()\[\]]+$", "", token).strip()
            if cleaned:  # Only add non-empty tokens
                cleaned_tokens.append(cleaned)
        
        # Rejoin tokens with a single space
        result = ' '.join(cleaned_tokens).strip()
        
        # Remove any remaining trailing punctuation from the final result
        result = re.sub(r"[.,;:()\[\]]+$", "", result).strip()
        
        # 8) safety: if result becomes empty, return ""
        if not result:
            return ""
        
        return result
    
    def _normalize_manufacturers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 4: Manufacturer normalization with M&A verification (web-verified, HIGH confidence only)."""
        # Use generalized column finder (not relying on column_map)
        manufacturer_col = self._find_manufacturer_column(df)
        
        if not manufacturer_col:
            print("WARNING: Manufacturer column not found. Skipping manufacturer normalization.")
            return df
        
        print(f"\n=== MANUFACTURER NORMALIZATION PHASE ===")
        print(f"Found manufacturer column: '{manufacturer_col}'")
        
        # Apply deterministic suffix cleanup first (in place)
        print("Applying deterministic suffix cleanup...")
        df[manufacturer_col] = df[manufacturer_col].apply(self._clean_manufacturer)
        
        # Then apply full normalization (M&A verification) if enabled
        # Get unique manufacturers for batch processing
        unique_manufacturers = df[manufacturer_col].dropna().unique()
        print(f"Processing {len(unique_manufacturers)} unique manufacturer names for canonicalization...")

        disable_web_verify = os.getenv('DISABLE_MANUFACTURER_WEB_VERIFY', '1') == '1'
        if disable_web_verify:
            print("Manufacturer web verification disabled. Using deterministic-only normalization.")
            manufacturer_map = self.manufacturer_normalizer.normalize_batch(
                list(unique_manufacturers),
                deterministic_only=True
            )
        else:
            manufacturer_map = self.manufacturer_normalizer.normalize_batch(
                list(unique_manufacturers),
                deterministic_only=False
            )
        
        # Apply mapping deterministically
        df[manufacturer_col] = df[manufacturer_col].map(manufacturer_map).fillna(df[manufacturer_col])

        # Capture sorted unique manufacturer list for audit report
        mfr_vals = [
            str(v).strip() for v in df[manufacturer_col].dropna().unique()
            if str(v).strip() and str(v).strip().lower() not in ('nan', 'none', '', 'null')
        ]
        self._audit_manufacturer_list = sorted(mfr_vals)

        print(f"Manufacturer normalization complete")

        return df
    
    def _extract_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 5: Extract important keywords from event text."""
        # Find event text column - try various patterns
        event_text_col = None
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            if 'event' in col_lower and 'text' in col_lower:
                event_text_col = col
                break
        
        # Extract keywords for each row
        keywords_list = []
        for idx, row in df.iterrows():
            if event_text_col:
                event_text = str(row[event_text_col]) if pd.notna(row[event_text_col]) else ""
            else:
                event_text = ""
            
            if not event_text or event_text.strip().lower() in ['nan', '', 'none', 'null']:
                keywords_list.append("")
            else:
                try:
                    keywords = self.groq_client.extract_keywords(event_text)
                    keywords_list.append(keywords)
                except Exception as e:
                    print(f"Warning: Error extracting keywords for row {idx}: {e}")
                    keywords_list.append("")
        
        # Check if column already exists
        if 'Important Keywords' in df.columns:
            # Update existing column
            df['Important Keywords'] = keywords_list
        else:
            # Add column immediately after Event Text (or at end if Event Text not found)
            if event_text_col:
                event_text_idx = df.columns.get_loc(event_text_col)
                df.insert(event_text_idx + 1, 'Important Keywords', keywords_list)
            else:
                df['Important Keywords'] = keywords_list
        
        return df
    
    def _map_imdrf_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 6: IMDRF code mapping (Deterministic primary + Groq fallback, Annex-controlled).

        Uses scalable batch processing for large files:
        - Files with 1000+ rows automatically use deterministic-only mode
        - Unique device problems are collected and processed once (not per-row)
        - Results are cached for efficiency
        """
        device_problem_col = self.column_map.get("device_problem")

        if device_problem_col is None:
            raise RuntimeError("HARD STOP: Device Problem column missing; cannot place IMDRF Code adjacent.")

        if device_problem_col not in df.columns:
            raise RuntimeError(f"HARD STOP: Device Problem column '{device_problem_col}' not found in dataframe.")

        print(f"Found Device Problem column: '{device_problem_col}' at position {df.columns.get_loc(device_problem_col)}")

        # Check if IMDRF mapper has loaded Annex
        total_codes = len(self.imdrf_mapper.level1_map) + len(self.imdrf_mapper.level2_map) + len(self.imdrf_mapper.level3_map)
        if total_codes == 0:
            print("WARNING: IMDRF lookup maps not loaded or empty. IMDRF codes will be blank.")
            # Still add the column, but fill with blanks
            if 'IMDRF Code' not in df.columns:
                cols = list(df.columns)
                insert_at = cols.index(device_problem_col) + 1
                cols.insert(insert_at, 'IMDRF Code')
                df = df.reindex(columns=cols)
                df['IMDRF Code'] = ""
            return df

        # Ensure IMDRF Code exists and is adjacent (insert only if missing)
        if 'IMDRF Code' not in df.columns:
            cols = list(df.columns)
            insert_at = cols.index(device_problem_col) + 1
            cols.insert(insert_at, 'IMDRF Code')
            df = df.reindex(columns=cols)
            df['IMDRF Code'] = ""
        else:
            # Hard-stop if not adjacent
            cols = list(df.columns)
            if cols.index('IMDRF Code') != cols.index(device_problem_col) + 1:
                raise RuntimeError("HARD STOP: IMDRF Code is not immediately right of Device Problem.")

        # Determine processing mode based on file size
        row_count = len(df)
        from backend.imdrf_mapper import LARGE_FILE_THRESHOLD

        print(f"Mapping IMDRF codes for {row_count} rows...")

        # Progress callback for console output
        def progress_callback(current, total, message):
            if current == 0 or current == total or current % 500 == 0:
                print(f"  [{current}/{total}] {message}")

        # Determine if we should use deterministic-only mode
        deterministic_only = row_count >= LARGE_FILE_THRESHOLD
        if deterministic_only:
            print(f"  Large file detected ({row_count} rows >= {LARGE_FILE_THRESHOLD}). Using FAST deterministic-only mode.")
            print(f"  Note: AI-assisted mapping disabled for performance. Some codes may be unmapped.")

        # Batch map unique device problems once
        device_problems = df[device_problem_col].tolist()
        mapping = self.imdrf_mapper.map_device_problems_batch(
            device_problems,
            deterministic_only=deterministic_only,
            progress_callback=progress_callback
        )

        # Explode rows so each device problem part becomes its own row
        df = self._explode_device_problem_rows(df, device_problem_col, mapping)
        
        # Validate all nonblank codes exist in Annex (handle multiple codes separated by ' | ')
        nonblank_cells = df[df['IMDRF Code'].notna() & (df['IMDRF Code'].astype(str).str.strip() != '')]['IMDRF Code'].unique()
        all_codes = set()
        for cell_value in nonblank_cells:
            cell_str = str(cell_value).strip()
            # Split on ' | ' to get individual codes
            codes = [c.strip() for c in cell_str.split(' | ') if c.strip()]
            all_codes.update(codes)
        invalid_codes = [code for code in all_codes if not self.imdrf_mapper.validate_code(code)]
        if invalid_codes:
            raise RuntimeError(f"HARD STOP: Found IMDRF codes not in Annex: {invalid_codes[:5]}")
        
        # Statistics
        non_empty = df[device_problem_col].notna() & (df[device_problem_col].astype(str).str.strip() != '')
        mapped = df['IMDRF Code'].notna() & (df['IMDRF Code'].astype(str).str.strip() != '')
        print(f"IMDRF mapping complete: {non_empty.sum()} non-empty device problems, {mapped.sum()} codes mapped")

        self._audit_imdrf_stats = {
            'non_empty_device_problems': int(non_empty.sum()),
            'mapped': int(mapped.sum()),
            'unmapped': int(non_empty.sum()) - int(mapped.sum()),
        }
        
        # Verify final position
        final_dp_idx = df.columns.get_loc(device_problem_col)
        final_imdrf_idx = df.columns.get_loc('IMDRF Code')
        if final_imdrf_idx != final_dp_idx + 1:
            raise RuntimeError(f"HARD STOP: IMDRF Code position verification failed! Device Problem: {final_dp_idx}, IMDRF Code: {final_imdrf_idx}")
        
        print(f"SUCCESS: IMDRF Code at position {final_imdrf_idx}, immediately after Device Problem at {final_dp_idx}")
        
        return df

    def _explode_device_problem_rows(self, df: pd.DataFrame, device_problem_col: str, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Expand rows so each semicolon-separated device problem becomes its own row
        with the mapped IMDRF Code beside it, while preserving all other columns.

        Vectorized to avoid materializing Python dicts per row (prevents OOM on
        large datasets).
        """
        raw_series = df[device_problem_col].fillna("").astype(str)
        parts_series = raw_series.apply(
            lambda s: [p.strip() for p in s.split(";") if p.strip()] or [s.strip()]
        )

        work = df.copy()
        work[device_problem_col] = parts_series
        exploded = work.explode(device_problem_col, ignore_index=True)
        exploded[device_problem_col] = exploded[device_problem_col].fillna("").astype(str)
        exploded['IMDRF Code'] = exploded[device_problem_col].map(lambda p: mapping.get(p, "") if p else "")

        # Preserve original column order
        return exploded[list(df.columns)]

    def _validate_output(self, df: pd.DataFrame, original_col_count: int) -> Dict[str, any]:
        """Phase 7: Final validation checks (HARD STOPS)."""
        results = {
            'column_count_correct': False,
            'no_timestamps': False,
            'date_format_correct': False,
            'imdrf_adjacent': False,
            'imdrf_codes_valid': False,
            'file_will_open': True,  # Assume true if we got here
            'all_passed': False,
            'details': {}
        }
        
        # Check 1: Only IMDRF Code added (if missing)
        new_cols = [col for col in df.columns if col in COLUMNS_TO_ADD]
        results['column_count_correct'] = len(new_cols) <= len(COLUMNS_TO_ADD) and all(col in df.columns for col in new_cols)
        results['details']['new_columns_found'] = new_cols
        results['details']['expected_columns'] = COLUMNS_TO_ADD
        
        # Check 2: No timestamps exist
        has_timestamps = False
        timestamp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower or 'timestamp' in col_lower:
                # Check if column contains time data
                sample = df[col].dropna().head(10)
                for val in sample:
                    val_str = str(val).strip()
                    if val_str and val_str.lower() != 'nan':
                        if ':' in val_str and len(val_str.split(':')) >= 2:
                            has_timestamps = True
                            timestamp_cols.append(col)
                            break
        results['no_timestamps'] = not has_timestamps
        results['details']['timestamp_columns'] = timestamp_cols
        
        # Check 3: Date format = DD-MM-YYYY (HARD STOP: no literal "nan", all nonblank dates must match format)
        event_date_col = self.column_map.get("event_date")
        date_received_col = self.column_map.get("date_received")
        date_columns = []
        if event_date_col and event_date_col in df.columns:
            date_columns.append(event_date_col)
        if date_received_col and date_received_col in df.columns:
            date_columns.append(date_received_col)
        
        date_format_correct = True
        date_issues = []
        for col in date_columns:
            # Check for literal "nan" (HARD STOP)
            has_nan = (df[col].astype(str).str.strip().str.lower() == 'nan').any()
            if has_nan:
                date_format_correct = False
                date_issues.append(f"{col}: contains literal 'nan'")
                break
            
            # Check format for all nonblank dates
            nonblank = df[col].dropna()
            for val in nonblank:
                val_str = str(val).strip()
                if val_str and val_str.lower() not in ['nan', '', 'none', 'null']:
                    if not re.match(r'^\d{2}-\d{2}-\d{4}$', val_str):
                        date_format_correct = False
                        date_issues.append(f"{col}: '{val_str[:20]}'")
                        break
        results['date_format_correct'] = date_format_correct
        results['details']['date_format_issues'] = date_issues[:5]
        
        # Check 4: IMDRF Code is adjacent to Device Problem (HARD STOP)
        device_problem_col = self.column_map.get("device_problem")
        imdrf_adjacent = False
        imdrf_position = None
        device_problem_position = None
        
        if 'IMDRF Code' in df.columns and device_problem_col and device_problem_col in df.columns:
            try:
                device_problem_idx = df.columns.get_loc(device_problem_col)
                imdrf_idx = df.columns.get_loc('IMDRF Code')
                device_problem_position = device_problem_idx
                imdrf_position = imdrf_idx
                
                # IMDRF Code MUST be immediately after Device Problem (position difference = 1)
                imdrf_adjacent = (imdrf_idx == device_problem_idx + 1)
            except (KeyError, IndexError):
                imdrf_adjacent = False
        elif device_problem_col is None:
            # Device Problem not found - this should have been caught earlier
            imdrf_adjacent = False
        
        results['imdrf_adjacent'] = imdrf_adjacent
        results['details']['imdrf_position'] = imdrf_position
        results['details']['device_problem_position'] = device_problem_position
        
        # Check 5: All nonblank IMDRF codes exist in Annex (HARD STOP) - handle multiple codes separated by ' | '
        imdrf_codes_valid = True
        if 'IMDRF Code' in df.columns:
            nonblank_cells = df[df['IMDRF Code'].notna() & (df['IMDRF Code'].astype(str).str.strip() != '')]['IMDRF Code'].unique()
            all_codes = set()
            for cell_value in nonblank_cells:
                cell_str = str(cell_value).strip()
                # Split on ' | ' to get individual codes
                codes = [c.strip() for c in cell_str.split(' | ') if c.strip()]
                all_codes.update(codes)
            invalid_codes = []
            for code in all_codes:
                if not self.imdrf_mapper.validate_code(code):
                    invalid_codes.append(code)
            if invalid_codes:
                imdrf_codes_valid = False
                results['details']['invalid_imdrf_codes'] = invalid_codes[:10]
        results['imdrf_codes_valid'] = imdrf_codes_valid
        
        results['details']['column_order'] = [f"{i}: {col}" for i, col in enumerate(df.columns)]
        
        results['all_passed'] = all([
            results['column_count_correct'],
            results['no_timestamps'],
            results['date_format_correct'],
            results['imdrf_adjacent'],
            results['imdrf_codes_valid'],
            results['file_will_open']
        ])
        
        return results
