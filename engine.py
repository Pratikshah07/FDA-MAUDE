"""Small wrapper providing a clean_maude_df(raw_df) function that uses MAUDEProcessor
to run the deterministic cleaning pipeline and return a cleaned DataFrame.
This writes temporary files and cleans up after itself.
"""
import os
import tempfile
import pandas as pd
from backend.processor import MAUDEProcessor


def clean_maude_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run MAUDEProcessor on an in-memory DataFrame and return the cleaned DataFrame.

    The implementation writes raw_df to a temporary CSV or Excel and calls
    MAUDEProcessor.process_file(input_path, output_path), and returns the cleaned
    DataFrame read back from the output Excel file.
    """
    tmpdir = tempfile.mkdtemp(prefix="maude_engine_")
    input_path = os.path.join(tmpdir, "input.csv")
    output_path = os.path.join(tmpdir, "cleaned_output.xlsx")
    try:
        # Write raw_df as CSV (deterministic text serialization)
        raw_df.to_csv(input_path, index=False, encoding="utf-8")

        processor = MAUDEProcessor()
        processor.process_file(input_path, output_path)

        cleaned_df = pd.read_excel(output_path, dtype=str, engine="openpyxl")
        # Ensure string trimming similar to existing pages
        for c in cleaned_df.columns:
            cleaned_df[c] = cleaned_df[c].astype(str).map(lambda v: v.strip()).replace({"nan": ""})
        return cleaned_df
    finally:
        # Best-effort cleanup
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        try:
            if os.path.exists(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)
        except Exception:
            pass