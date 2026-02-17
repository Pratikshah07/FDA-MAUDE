"""
TXT to CSV Converter Module

This module provides functionality to convert large TXT files (up to 1GB+) to CSV format.
It uses chunked processing to handle large files efficiently without loading entire file into memory.

Supports common delimiters: pipe (|), tab, comma, semicolon
Validates data integrity during conversion.
"""

import os
import csv
import re
from typing import Optional, Dict, List, Tuple, Generator
from datetime import datetime


class TxtToCsvConverter:
    """
    Converts large TXT files to CSV format with validation.

    Features:
    - Chunked processing for memory efficiency
    - Auto-detection of delimiters
    - Data validation
    - Progress tracking
    - Error handling for malformed rows
    """

    # Common delimiters in order of likelihood for MAUDE data
    DELIMITERS = ['|', '\t', ',', ';']

    # Chunk size for reading (64KB chunks)
    CHUNK_SIZE = 64 * 1024

    # Maximum rows to sample for delimiter detection
    SAMPLE_ROWS = 100

    def __init__(self):
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'empty_rows': 0,
            'columns_detected': 0,
            'delimiter_used': '',
            'file_size_bytes': 0,
            'processing_time_seconds': 0,
            'errors': []
        }

    def detect_delimiter(self, file_path: str) -> str:
        """
        Auto-detect the delimiter used in the TXT file.

        Args:
            file_path: Path to the TXT file

        Returns:
            Detected delimiter character
        """
        delimiter_counts = {d: [] for d in self.DELIMITERS}

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if i >= self.SAMPLE_ROWS:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    for delimiter in self.DELIMITERS:
                        count = line.count(delimiter)
                        delimiter_counts[delimiter].append(count)

        except Exception as e:
            # Default to pipe if detection fails
            self.stats['errors'].append(f"Delimiter detection error: {str(e)}")
            return '|'

        # Find delimiter with most consistent non-zero count
        best_delimiter = '|'
        best_score = -1

        for delimiter, counts in delimiter_counts.items():
            if not counts:
                continue

            # Filter out zero counts
            non_zero = [c for c in counts if c > 0]
            if not non_zero:
                continue

            # Calculate consistency score (prefer consistent counts)
            avg_count = sum(non_zero) / len(non_zero)
            if avg_count < 1:
                continue

            # Score based on count and consistency
            variance = sum((c - avg_count) ** 2 for c in non_zero) / len(non_zero)
            consistency = 1 / (1 + variance)
            score = avg_count * consistency * len(non_zero)

            if score > best_score:
                best_score = score
                best_delimiter = delimiter

        return best_delimiter

    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding by reading first few bytes.

        Args:
            file_path: Path to the file

        Returns:
            Encoding string (utf-8, latin-1, etc.)
        """
        # Try common encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # Try to read first 10KB
                    f.read(10240)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Default to latin-1 (accepts all byte values)
        return 'latin-1'

    def count_columns(self, file_path: str, delimiter: str, encoding: str) -> int:
        """
        Count the number of columns in the file.

        Args:
            file_path: Path to the file
            delimiter: Delimiter character
            encoding: File encoding

        Returns:
            Number of columns detected
        """
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                # Read first non-empty line (header)
                for line in f:
                    line = line.strip()
                    if line:
                        return len(line.split(delimiter))
        except Exception:
            pass

        return 0

    def validate_row(self, row: List[str], expected_columns: int, row_number: int) -> Tuple[bool, str]:
        """
        Validate a single row of data.

        Args:
            row: List of field values
            expected_columns: Expected number of columns
            row_number: Row number for error reporting

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check column count
        if len(row) != expected_columns:
            return False, f"Row {row_number}: Expected {expected_columns} columns, got {len(row)}"

        # Check for completely empty rows (all fields empty)
        if all(not field.strip() for field in row):
            return False, f"Row {row_number}: All fields are empty"

        return True, ""

    def clean_field(self, field: str) -> str:
        """
        Clean a single field value.

        Args:
            field: Raw field value

        Returns:
            Cleaned field value
        """
        # Strip whitespace
        field = field.strip()

        # Remove surrounding quotes if present
        if len(field) >= 2:
            if (field[0] == '"' and field[-1] == '"') or \
               (field[0] == "'" and field[-1] == "'"):
                field = field[1:-1]

        # Replace literal "nan" or "NULL" with empty string
        if field.lower() in ['nan', 'null', 'none', 'n/a', 'na']:
            return ''

        return field

    def process_file_chunked(self, input_path: str, output_path: str,
                             delimiter: Optional[str] = None,
                             progress_callback=None) -> Dict:
        """
        Process large TXT file and convert to CSV using chunked reading.

        Args:
            input_path: Path to input TXT file
            output_path: Path to output CSV file
            delimiter: Optional delimiter (auto-detected if not provided)
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary with conversion statistics
        """
        start_time = datetime.now()

        # Reset stats
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'empty_rows': 0,
            'columns_detected': 0,
            'delimiter_used': '',
            'file_size_bytes': 0,
            'processing_time_seconds': 0,
            'errors': [],
            'sample_errors': []  # Store first 10 errors as samples
        }

        # Get file size
        self.stats['file_size_bytes'] = os.path.getsize(input_path)

        # Detect encoding
        encoding = self.detect_encoding(input_path)
        print(f"[TXT-CSV] Detected encoding: {encoding}")

        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = self.detect_delimiter(input_path)

        self.stats['delimiter_used'] = repr(delimiter)
        print(f"[TXT-CSV] Using delimiter: {repr(delimiter)}")

        # Count expected columns from header
        expected_columns = self.count_columns(input_path, delimiter, encoding)
        self.stats['columns_detected'] = expected_columns
        print(f"[TXT-CSV] Detected {expected_columns} columns")

        if expected_columns == 0:
            self.stats['errors'].append("Could not detect columns in file")
            return self.stats

        bytes_processed = 0
        rows_written = 0

        try:
            with open(input_path, 'r', encoding=encoding, errors='replace') as infile, \
                 open(output_path, 'w', newline='', encoding='utf-8-sig') as outfile:
                # Use utf-8-sig to add BOM for Excel compatibility on Windows
                # Use QUOTE_ALL to ensure proper CSV formatting
                csv_writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

                for line_number, line in enumerate(infile, 1):
                    # Track progress
                    bytes_processed += len(line.encode('utf-8', errors='replace'))

                    # Progress callback every 10000 rows
                    if progress_callback and line_number % 10000 == 0:
                        progress = min(100, int((bytes_processed / self.stats['file_size_bytes']) * 100))
                        progress_callback(progress, line_number)

                    # Skip empty lines
                    stripped_line = line.strip()
                    if not stripped_line:
                        self.stats['empty_rows'] += 1
                        continue

                    self.stats['total_rows'] += 1

                    # Split line by delimiter
                    fields = stripped_line.split(delimiter)

                    # Clean each field
                    cleaned_fields = [self.clean_field(f) for f in fields]

                    # Debug first few rows
                    if line_number <= 3:
                        print(f"[TXT-CSV] Row {line_number}: {len(cleaned_fields)} fields")

                    # Validate row
                    is_valid, error_msg = self.validate_row(cleaned_fields, expected_columns, line_number)

                    if is_valid:
                        csv_writer.writerow(cleaned_fields)
                        rows_written += 1
                        self.stats['valid_rows'] += 1
                    else:
                        self.stats['invalid_rows'] += 1

                        # Store sample errors (first 10)
                        if len(self.stats['sample_errors']) < 10:
                            self.stats['sample_errors'].append(error_msg)

                        # Handle rows with wrong column count
                        # Pad or truncate to match expected columns
                        if len(cleaned_fields) < expected_columns:
                            cleaned_fields.extend([''] * (expected_columns - len(cleaned_fields)))
                        elif len(cleaned_fields) > expected_columns:
                            # Merge extra fields into last column
                            extra = delimiter.join(cleaned_fields[expected_columns-1:])
                            cleaned_fields = cleaned_fields[:expected_columns-1] + [extra]

                        csv_writer.writerow(cleaned_fields)
                        rows_written += 1
                        self.stats['valid_rows'] += 1  # Count as valid after fixing
                        self.stats['invalid_rows'] -= 1

            print(f"[TXT-CSV] Wrote {rows_written} rows to CSV")

        except Exception as e:
            self.stats['errors'].append(f"Processing error: {str(e)}")
            print(f"[TXT-CSV] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time_seconds'] = (end_time - start_time).total_seconds()

        # Final progress callback
        if progress_callback:
            progress_callback(100, self.stats['total_rows'])

        # Verify output file
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            print(f"[TXT-CSV] Output file size: {output_size} bytes")
            self.stats['output_file_size'] = output_size

        return self.stats

    def get_file_preview(self, file_path: str, num_rows: int = 10) -> Dict:
        """
        Get a preview of the file contents for validation.

        Args:
            file_path: Path to the file
            num_rows: Number of rows to preview

        Returns:
            Dictionary with preview data
        """
        preview = {
            'headers': [],
            'rows': [],
            'delimiter': '',
            'encoding': '',
            'total_columns': 0,
            'file_size_mb': 0,
            'estimated_rows': 0
        }

        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            preview['file_size_mb'] = round(file_size / (1024 * 1024), 2)

            # Detect encoding and delimiter
            encoding = self.detect_encoding(file_path)
            delimiter = self.detect_delimiter(file_path)
            preview['encoding'] = encoding
            preview['delimiter'] = repr(delimiter)

            # Read sample rows
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                line_count = 0
                total_line_length = 0

                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    line_count += 1
                    total_line_length += len(line)

                    fields = [self.clean_field(f) for f in line.split(delimiter)]

                    if i == 0:
                        preview['headers'] = fields
                        preview['total_columns'] = len(fields)
                    elif len(preview['rows']) < num_rows:
                        preview['rows'].append(fields)

                    if line_count >= num_rows + 1:
                        break

                # Estimate total rows based on average line length
                if total_line_length > 0 and line_count > 0:
                    avg_line_length = total_line_length / line_count
                    preview['estimated_rows'] = int(file_size / avg_line_length)

        except Exception as e:
            preview['error'] = str(e)

        return preview


def convert_txt_to_csv(input_path: str, output_path: str,
                       delimiter: Optional[str] = None) -> Dict:
    """
    Convenience function to convert TXT to CSV.

    Args:
        input_path: Path to input TXT file
        output_path: Path to output CSV file
        delimiter: Optional delimiter character

    Returns:
        Conversion statistics dictionary
    """
    converter = TxtToCsvConverter()
    return converter.process_file_chunked(input_path, output_path, delimiter)


def get_txt_preview(file_path: str, num_rows: int = 10) -> Dict:
    """
    Convenience function to get file preview.

    Args:
        file_path: Path to the file
        num_rows: Number of rows to preview

    Returns:
        Preview dictionary
    """
    converter = TxtToCsvConverter()
    return converter.get_file_preview(file_path, num_rows)
