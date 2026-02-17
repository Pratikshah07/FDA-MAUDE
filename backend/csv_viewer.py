"""
CSV Viewer Module

This module provides functionality to view large CSV files (10GB+) with pagination.
It uses streaming to avoid loading entire file into memory.
"""

import os
import csv
import mmap
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class LargeCSVViewer:
    """
    Viewer for large CSV files with pagination support.

    Features:
    - Memory-efficient streaming for 10GB+ files
    - Pagination with configurable page size
    - Column statistics and data validation
    - Search functionality
    """

    # Default page size
    DEFAULT_PAGE_SIZE = 100

    # Maximum columns to display (for very wide files)
    MAX_DISPLAY_COLUMNS = 50

    def __init__(self):
        self.file_info = {}

    def detect_delimiter(self, file_path: str, sample_size: int = 10240) -> str:
        """
        Detect the delimiter used in the CSV file.

        Args:
            file_path: Path to the CSV file
            sample_size: Number of bytes to sample

        Returns:
            Detected delimiter character
        """
        delimiters = [',', '|', '\t', ';']
        delimiter_counts = {d: 0 for d in delimiters}

        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                sample = f.read(sample_size)
                lines = sample.split('\n')[:20]  # Check first 20 lines

                for line in lines:
                    if not line.strip():
                        continue
                    for d in delimiters:
                        delimiter_counts[d] += line.count(d)

            # Return delimiter with highest count
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            if delimiter_counts[best_delimiter] > 0:
                return best_delimiter

        except Exception:
            pass

        return ','  # Default to comma

    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding.

        Args:
            file_path: Path to the file

        Returns:
            Encoding string
        """
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(8192)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        return 'latin-1'

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get basic file information without reading entire file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary with file information
        """
        info = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size_bytes': 0,
            'file_size_mb': 0,
            'total_rows': 0,
            'total_columns': 0,
            'headers': [],
            'delimiter': ',',
            'encoding': 'utf-8',
            'sample_row': []
        }

        try:
            # Get file size
            info['file_size_bytes'] = os.path.getsize(file_path)
            info['file_size_mb'] = round(info['file_size_bytes'] / (1024 * 1024), 2)

            # Detect encoding and delimiter
            info['encoding'] = self.detect_encoding(file_path)
            info['delimiter'] = self.detect_delimiter(file_path)

            # Count rows and get headers
            with open(file_path, 'r', encoding=info['encoding'], errors='replace') as f:
                # Read header
                header_line = f.readline().strip()
                if header_line:
                    info['headers'] = header_line.split(info['delimiter'])
                    # Clean headers - remove quotes
                    info['headers'] = [h.strip().strip('"\'') for h in info['headers']]
                    info['total_columns'] = len(info['headers'])

                # Read first data row as sample
                sample_line = f.readline().strip()
                if sample_line:
                    info['sample_row'] = sample_line.split(info['delimiter'])
                    info['sample_row'] = [s.strip().strip('"\'') for s in info['sample_row']]

                # Estimate row count based on file size and sample
                # Count actual rows for smaller files, estimate for large ones
                if info['file_size_bytes'] < 100 * 1024 * 1024:  # Less than 100MB
                    row_count = 1  # Header
                    for _ in f:
                        row_count += 1
                    info['total_rows'] = row_count
                else:
                    # Estimate based on average line length
                    f.seek(0)
                    sample_bytes = 0
                    sample_lines = 0
                    for i, line in enumerate(f):
                        if i >= 1000:
                            break
                        sample_bytes += len(line.encode('utf-8', errors='replace'))
                        sample_lines += 1

                    if sample_lines > 0:
                        avg_line_size = sample_bytes / sample_lines
                        info['total_rows'] = int(info['file_size_bytes'] / avg_line_size)
                    else:
                        info['total_rows'] = 0

        except Exception as e:
            info['error'] = str(e)

        self.file_info = info
        return info

    def get_page(self, file_path: str, page: int = 1, page_size: int = None,
                 delimiter: str = None, encoding: str = None) -> Dict:
        """
        Get a specific page of data from the CSV file.

        Args:
            file_path: Path to the CSV file
            page: Page number (1-indexed)
            page_size: Number of rows per page
            delimiter: CSV delimiter (auto-detected if None)
            encoding: File encoding (auto-detected if None)

        Returns:
            Dictionary with page data
        """
        if page_size is None:
            page_size = self.DEFAULT_PAGE_SIZE

        result = {
            'page': page,
            'page_size': page_size,
            'total_pages': 0,
            'total_rows': 0,
            'headers': [],
            'rows': [],
            'start_row': 0,
            'end_row': 0,
            'has_next': False,
            'has_prev': page > 1
        }

        try:
            # Detect encoding and delimiter if not provided
            if encoding is None:
                encoding = self.detect_encoding(file_path)
            if delimiter is None:
                delimiter = self.detect_delimiter(file_path)

            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                # Read and parse header
                header_line = f.readline().strip()
                if header_line:
                    headers = header_line.split(delimiter)
                    result['headers'] = [h.strip().strip('"\'') for h in headers]

                # Calculate row positions
                start_row = (page - 1) * page_size
                end_row = start_row + page_size

                result['start_row'] = start_row + 1  # 1-indexed for display

                # Skip to start position
                current_row = 0
                for _ in range(start_row):
                    line = f.readline()
                    if not line:
                        break
                    current_row += 1

                # Read page data
                rows_read = 0
                total_rows = start_row  # Start counting from where we are

                for line in f:
                    total_rows += 1

                    if rows_read < page_size:
                        line = line.strip()
                        if line:
                            fields = line.split(delimiter)
                            # Clean fields - remove quotes
                            cleaned = [f.strip().strip('"\'') for f in fields]

                            # Ensure consistent column count
                            if len(cleaned) < len(result['headers']):
                                cleaned.extend([''] * (len(result['headers']) - len(cleaned)))
                            elif len(cleaned) > len(result['headers']):
                                cleaned = cleaned[:len(result['headers'])]

                            result['rows'].append(cleaned)
                            rows_read += 1

                # Count remaining rows for total (sample-based for large files)
                remaining_sample = 0
                for i, _ in enumerate(f):
                    remaining_sample += 1
                    if i >= 10000:  # Sample 10000 more lines
                        # Estimate remaining based on file position
                        current_pos = f.tell()
                        file_size = os.path.getsize(file_path)
                        remaining_ratio = (file_size - current_pos) / current_pos if current_pos > 0 else 0
                        remaining_sample = int(remaining_sample * (1 + remaining_ratio))
                        break

                total_rows += remaining_sample

                result['total_rows'] = total_rows
                result['end_row'] = min(start_row + rows_read, total_rows)
                result['total_pages'] = max(1, (total_rows + page_size - 1) // page_size)
                result['has_next'] = page < result['total_pages']

        except Exception as e:
            result['error'] = str(e)
            import traceback
            traceback.print_exc()

        return result

    def search_in_file(self, file_path: str, search_term: str,
                       max_results: int = 100, encoding: str = None,
                       delimiter: str = None) -> Dict:
        """
        Search for a term in the CSV file.

        Args:
            file_path: Path to the CSV file
            search_term: Term to search for
            max_results: Maximum number of results to return
            encoding: File encoding
            delimiter: CSV delimiter

        Returns:
            Dictionary with search results
        """
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        if delimiter is None:
            delimiter = self.detect_delimiter(file_path)

        results = {
            'search_term': search_term,
            'total_matches': 0,
            'matches': [],
            'headers': []
        }

        try:
            search_lower = search_term.lower()

            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                # Read header
                header_line = f.readline().strip()
                if header_line:
                    results['headers'] = [h.strip().strip('"\'') for h in header_line.split(delimiter)]

                # Search through file
                for row_num, line in enumerate(f, start=2):  # Start at 2 (after header)
                    if search_lower in line.lower():
                        fields = line.strip().split(delimiter)
                        cleaned = [f.strip().strip('"\'') for f in fields]

                        results['matches'].append({
                            'row_number': row_num,
                            'data': cleaned
                        })
                        results['total_matches'] += 1

                        if len(results['matches']) >= max_results:
                            break

        except Exception as e:
            results['error'] = str(e)

        return results

    def get_column_stats(self, file_path: str, column_index: int,
                         sample_size: int = 10000, encoding: str = None,
                         delimiter: str = None) -> Dict:
        """
        Get statistics for a specific column.

        Args:
            file_path: Path to the CSV file
            column_index: Index of column to analyze
            sample_size: Number of rows to sample
            encoding: File encoding
            delimiter: CSV delimiter

        Returns:
            Dictionary with column statistics
        """
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        if delimiter is None:
            delimiter = self.detect_delimiter(file_path)

        stats = {
            'column_index': column_index,
            'column_name': '',
            'total_sampled': 0,
            'non_empty_count': 0,
            'empty_count': 0,
            'unique_values': 0,
            'sample_values': [],
            'numeric_count': 0,
            'min_length': float('inf'),
            'max_length': 0
        }

        try:
            unique_values = set()
            sample_values = []

            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                # Read header
                header_line = f.readline().strip()
                if header_line:
                    headers = [h.strip().strip('"\'') for h in header_line.split(delimiter)]
                    if column_index < len(headers):
                        stats['column_name'] = headers[column_index]

                # Sample rows
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break

                    fields = line.strip().split(delimiter)
                    if column_index < len(fields):
                        value = fields[column_index].strip().strip('"\'')
                        stats['total_sampled'] += 1

                        if value:
                            stats['non_empty_count'] += 1
                            unique_values.add(value)

                            stats['min_length'] = min(stats['min_length'], len(value))
                            stats['max_length'] = max(stats['max_length'], len(value))

                            # Check if numeric
                            try:
                                float(value.replace(',', ''))
                                stats['numeric_count'] += 1
                            except ValueError:
                                pass

                            # Collect sample values
                            if len(sample_values) < 20:
                                if value not in sample_values:
                                    sample_values.append(value)
                        else:
                            stats['empty_count'] += 1

            stats['unique_values'] = len(unique_values)
            stats['sample_values'] = sample_values

            if stats['min_length'] == float('inf'):
                stats['min_length'] = 0

        except Exception as e:
            stats['error'] = str(e)

        return stats


def get_csv_page(file_path: str, page: int = 1, page_size: int = 100) -> Dict:
    """
    Convenience function to get a page of CSV data.

    Args:
        file_path: Path to CSV file
        page: Page number
        page_size: Rows per page

    Returns:
        Page data dictionary
    """
    viewer = LargeCSVViewer()
    return viewer.get_page(file_path, page, page_size)


def get_csv_info(file_path: str) -> Dict:
    """
    Convenience function to get CSV file info.

    Args:
        file_path: Path to CSV file

    Returns:
        File info dictionary
    """
    viewer = LargeCSVViewer()
    return viewer.get_file_info(file_path)
