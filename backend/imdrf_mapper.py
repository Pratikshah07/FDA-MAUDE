"""
Deterministic IMDRF mapping with Groq fallback (Annex-controlled).

Scalability Features:
- Batch processing for Groq API calls (reduces API calls by processing unique problems together)
- Deterministic-only mode for large files (skip Groq entirely for fast processing)
- Progress callback support for real-time feedback
- Aggressive caching to avoid repeated API calls
"""
import json
import os
import pandas as pd
import re
import time
from typing import Dict, Optional, List, Callable, Set
from backend.groq_client import GroqClient


# Threshold for deterministic-only mode (rows)
LARGE_FILE_THRESHOLD = 1000
# Batch size for Groq API calls
GROQ_BATCH_SIZE = 10
# Rate limit delay between batches (seconds)
GROQ_RATE_LIMIT_DELAY = 0.5


class IMDRFMapper:
    """Maps Device Problem to IMDRF codes using deterministic matching with Groq fallback."""
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        try:
            self.groq_client = groq_client if groq_client is not None else GroqClient()
        except Exception as e:
            print(f"Warning: Could not initialize GroqClient: {e}. IMDRF mapping will use deterministic methods only.")
            self.groq_client = None
        self.level1_map = {}
        self.level2_map = {}
        self.level3_map = {}
        self.level1_terms = []  # For Groq fallback
        self.level2_hierarchy = {}  # level1_term -> [level2_terms]
        self.level3_hierarchy = {}  # level2_term -> [level3_terms]
        # Use /tmp on Vercel (serverless) or cache/ for local
        if os.path.exists('/tmp'):
            self.cache_dir = os.path.join('/tmp', 'maude_cache')
        elif os.getenv('TMPDIR'):
            self.cache_dir = os.path.join(os.getenv('TMPDIR'), 'maude_cache')
        elif os.getenv('TMP'):
            self.cache_dir = os.path.join(os.getenv('TMP'), 'maude_cache')
        else:
            self.cache_dir = os.path.join('cache', 'maude_cache')
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            # Fallback to /tmp if available
            if self.cache_dir != os.path.join('/tmp', 'maude_cache') and os.path.exists('/tmp'):
                self.cache_dir = os.path.join('/tmp', 'maude_cache')
                os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "device_problem_to_imdrf_cache.json")
        self.cache = self._load_cache()
        self.annex_codes = set()  # All valid codes for validation
    
    def _load_cache(self) -> dict:
        """Load IMDRF mapping cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save IMDRF mapping cache."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save IMDRF cache: {e}")
    
    def _norm_term(self, s: str) -> str:
        """Deterministic normalization for matching only."""
        if s is None:
            return ""
        s = str(s).strip().lower()
        if not s or s == "nan":
            return ""
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*/\s*", "/", s)
        s = re.sub(r"[.,;:]+$", "", s).strip()
        return s
    
    def _is_blank(self, x) -> bool:
        if x is None:
            return True
        s = str(x).strip()
        return (s == "") or (s.lower() == "nan")
    
    def load_annex(self, annex_xlsx_path: str):
        """Load Annex structure deterministically."""
        xl = pd.ExcelFile(annex_xlsx_path)
        
        level1_map, level2_map, level3_map = {}, {}, {}
        level1_terms = []
        level2_hierarchy = {}
        level3_hierarchy = {}
        annex_codes = set()
        
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
            
            current_l1 = None
            current_l2 = None
            
            for _, r in df.iterrows():
                code = "" if r.get("Code") is None else str(r.get("Code")).strip()
                if not code or code.lower() == "nan":
                    continue
                
                annex_codes.add(code)
                
                l1 = self._norm_term(r.get("Level 1 Term"))
                l2 = self._norm_term(r.get("Level 2 Term"))
                l3 = self._norm_term(r.get("Level 3 Term"))
                
                if len(code) == 3 and l1:
                    level1_map.setdefault(l1, code)
                    if l1 not in level1_terms:
                        level1_terms.append(l1)
                    current_l1 = l1
                elif len(code) == 5 and l2:
                    level2_map.setdefault(l2, code)
                    if current_l1:
                        if current_l1 not in level2_hierarchy:
                            level2_hierarchy[current_l1] = []
                        if l2 not in level2_hierarchy[current_l1]:
                            level2_hierarchy[current_l1].append(l2)
                    current_l2 = l2
                elif len(code) == 7 and l3:
                    level3_map.setdefault(l3, code)
                    if current_l2:
                        if current_l2 not in level3_hierarchy:
                            level3_hierarchy[current_l2] = []
                        if l3 not in level3_hierarchy[current_l2]:
                            level3_hierarchy[current_l2].append(l3)
        
        self.level1_map = level1_map
        self.level2_map = level2_map
        self.level3_map = level3_map
        self.level1_terms = level1_terms
        self.level2_hierarchy = level2_hierarchy
        self.level3_hierarchy = level3_hierarchy
        self.annex_codes = annex_codes
        
        print(f"Loaded Annex: L1={len(level1_map)}, L2={len(level2_map)}, L3={len(level3_map)}, Total codes={len(annex_codes)}")
    
    def _map_one_problem_to_code(self, problem_part: str) -> str:
        """
        Map a single Device Problem part to one IMDRF code.
        Deterministic: exact match only after normalization.
        Preference: level3 > level2 > level1.
        """
        pn = self._norm_term(problem_part)
        if not pn:
            return ""
        if pn == self._norm_term("Appropriate Term/Code Not Available"):
            return ""

        if pn in self.level3_map:
            return self.level3_map[pn]
        if pn in self.level2_map:
            return self.level2_map[pn]
        if pn in self.level1_map:
            return self.level1_map[pn]
        return ""

    def _deterministic_map(self, device_problem_part: str) -> str:
        """Deterministic mapping for a single part."""
        if self._is_blank(device_problem_part):
            return ""
        
        raw = str(device_problem_part).strip()
        if not raw:
            return ""
        
        if self._norm_term(raw) == self._norm_term("Appropriate Term/Code Not Available"):
            return ""
        
        pn = self._norm_term(raw)
        if not pn:
            return ""
        
        # Check cache first
        if pn in self.cache:
            return self.cache[pn]
        
        # Try deterministic matching
        code = None
        if pn in self.level3_map:
            code = self.level3_map[pn]
        elif pn in self.level2_map:
            code = self.level2_map[pn]
        elif pn in self.level1_map:
            code = self.level1_map[pn]
        
        # Cache result (even if blank)
        self.cache[pn] = code if code else ""
        self._save_cache()
        
        return code if code else ""
    
    def _groq_fallback_level1(self, device_problem_part: str) -> Optional[str]:
        """Groq fallback for Level-1 selection (Annex-controlled)."""
        if not self.groq_client or not self.groq_client.available:
            return None
        terms_str = ', '.join([f'"{t}"' for t in self.level1_terms[:100]])  # Limit to avoid token limits
        
        prompt = f"""You are selecting the most appropriate Level-1 IMDRF term for a device problem.

Device Problem: "{device_problem_part}"

Available Level-1 terms (from Annex, you MUST select exactly one from this list):
{terms_str}

CRITICAL RULES:
- Return ONLY valid JSON
- Select the EXACT term from the list above that best matches the device problem
- If no good match, return {{"selected": "NO_MATCH"}}
- Do NOT modify or paraphrase the term

Return format (JSON only):
{{"selected": "<exact term from list>"}} OR {{"selected": "NO_MATCH"}}"""

        try:
            response = self.groq_client.client.chat.completions.create(
                model=self.groq_client.model,
                messages=[
                    {"role": "system", "content": "You are an IMDRF coding expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            selected = result.get("selected", "NO_MATCH")
            
            if selected == "NO_MATCH":
                return None
            
            # Validate: must be in level1_terms
            selected_norm = self._norm_term(selected)
            if selected_norm in self.level1_map:
                return self.level1_map[selected_norm]
            
            return None
            
        except Exception as e:
            print(f"Warning: Groq Level-1 selection failed: {e}")
            return None
    
    def _groq_fallback_level2(self, device_problem_part: str, level1_term: str) -> Optional[str]:
        """Groq fallback for Level-2 selection (Annex-controlled)."""
        if not self.groq_client or not self.groq_client.available:
            return None
        level2_terms = self.level2_hierarchy.get(level1_term, [])
        if not level2_terms:
            return None
        
        terms_str = ', '.join([f'"{t}"' for t in level2_terms[:50]])
        
        prompt = f"""You are selecting the most appropriate Level-2 IMDRF term for a device problem.

Device Problem: "{device_problem_part}"
Level-1 context: "{level1_term}"

Available Level-2 terms (under Level-1, you MUST select exactly one from this list):
{terms_str}

CRITICAL RULES:
- Return ONLY valid JSON
- Select the EXACT term from the list above
- If no good match, return {{"selected": "NO_MATCH"}}

Return format (JSON only):
{{"selected": "<exact term>"}} OR {{"selected": "NO_MATCH"}}"""

        try:
            response = self.groq_client.client.chat.completions.create(
                model=self.groq_client.model,
                messages=[
                    {"role": "system", "content": "You are an IMDRF coding expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            selected = result.get("selected", "NO_MATCH")
            
            if selected == "NO_MATCH":
                return None
            
            selected_norm = self._norm_term(selected)
            if selected_norm in self.level2_map:
                return self.level2_map[selected_norm]
            
            return None
            
        except Exception as e:
            print(f"Warning: Groq Level-2 selection failed: {e}")
            return None
    
    def _groq_fallback_level3(self, device_problem_part: str, level2_term: str) -> Optional[str]:
        """Groq fallback for Level-3 selection (Annex-controlled)."""
        level3_terms = self.level3_hierarchy.get(level2_term, [])
        if not level3_terms:
            return None
        
        terms_str = ', '.join([f'"{t}"' for t in level3_terms[:50]])
        
        prompt = f"""You are selecting the most appropriate Level-3 IMDRF term for a device problem.

Device Problem: "{device_problem_part}"
Level-2 context: "{level2_term}"

Available Level-3 terms (under Level-2, you MUST select exactly one from this list):
{terms_str}

CRITICAL RULES:
- Return ONLY valid JSON
- Select the EXACT term from the list above
- If no good match, return {{"selected": "NO_MATCH"}}

Return format (JSON only):
{{"selected": "<exact term>"}} OR {{"selected": "NO_MATCH"}}"""

        try:
            response = self.groq_client.client.chat.completions.create(
                model=self.groq_client.model,
                messages=[
                    {"role": "system", "content": "You are an IMDRF coding expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            selected = result.get("selected", "NO_MATCH")
            
            if selected == "NO_MATCH":
                return None
            
            selected_norm = self._norm_term(selected)
            if selected_norm in self.level3_map:
                return self.level3_map[selected_norm]
            
            return None
            
        except Exception as e:
            print(f"Warning: Groq Level-3 selection failed: {e}")
            return None
    
    def _groq_fallback(self, device_problem_part: str) -> str:
        """Groq fallback hierarchical selection (Annex-controlled)."""
        pn = self._norm_term(device_problem_part)
        if not pn:
            return ""
        
        # Try Level-1
        l1_code = self._groq_fallback_level1(device_problem_part)
        if not l1_code:
            return ""
        
        # Try Level-2
        l1_term_norm = None
        for term, code in self.level1_map.items():
            if code == l1_code:
                l1_term_norm = term
                break
        
        if not l1_term_norm:
            return l1_code  # Return Level-1 code
        
        l2_code = self._groq_fallback_level2(device_problem_part, l1_term_norm)
        if not l2_code:
            return l1_code  # Return Level-1 code
        
        # Try Level-3
        l2_term_norm = None
        for term, code in self.level2_map.items():
            if code == l2_code:
                l2_term_norm = term
                break
        
        if not l2_term_norm:
            return l2_code  # Return Level-2 code
        
        l3_code = self._groq_fallback_level3(device_problem_part, l2_term_norm)
        if l3_code:
            return l3_code  # Return Level-3 code
        
        return l2_code  # Return Level-2 code
    
    def map_device_problem_cell_to_codes(self, device_problem_value: str) -> str:
        """
        Handles multi-problem cells:
          - split on ';'
          - map each part independently
          - output codes joined by ' | ' in original order
          - de-duplicate codes while preserving order
          - skip unmapped parts
        """
        if device_problem_value is None:
            return ""
        raw = str(device_problem_value).strip()
        if not raw or raw.lower() == "nan":
            return ""

        # Split multiple problems
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        if not parts:
            return ""

        codes_in_order = []
        seen = set()

        for part in parts:
            code = self._map_one_problem_to_code(part)
            if not code:
                continue  # fail-safe: skip unmapped part
            if code not in seen:
                seen.add(code)
                codes_in_order.append(code)

        return " | ".join(codes_in_order) if codes_in_order else ""

    def map_device_problem(self, device_problem_value: str) -> str:
        """
        Map Device Problem to IMDRF code.
        Uses deterministic matching first, Groq fallback if needed.
        """
        if self._is_blank(device_problem_value):
            return ""
        
        raw = str(device_problem_value).strip()
        if not raw:
            return ""
        
        if self._norm_term(raw) == self._norm_term("Appropriate Term/Code Not Available"):
            return ""
        
        # Split on semicolons
        parts = [p.strip() for p in raw.split(";")]
        codes = []
        
        for p in parts:
            if self._is_blank(p):
                continue
            
            # Try deterministic first
            code = self._deterministic_map(p)
            
            # Groq fallback if deterministic failed
            if not code:
                pn = self._norm_term(p)
                # Only use Groq if not in cache (to avoid repeated calls)
                if pn and pn not in self.cache:
                    code = self._groq_fallback(p)
                    # Cache the result
                    self.cache[pn] = code
                    self._save_cache()
            
            if code:
                codes.append(code)
        
        if not codes:
            return ""
        
        # Choose deepest code (len 7 > 5 > 3), tie-breaker lexicographically smallest
        codes_sorted = sorted(codes, key=lambda c: (-len(c), c))
        return codes_sorted[0]
    
    def validate_code(self, code: str) -> bool:
        """Validate that code exists in Annex."""
        return code in self.annex_codes

    # ==================== SCALABILITY METHODS ====================

    def map_device_problems_batch(
        self,
        device_problems: List[str],
        deterministic_only: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, str]:
        """
        Batch process multiple device problems efficiently.

        This method:
        1. Collects unique problems
        2. Applies deterministic mapping first (fast)
        3. Batches Groq API calls for remaining problems (if not deterministic_only)
        4. Caches all results for future use

        Args:
            device_problems: List of device problem strings
            deterministic_only: If True, skip all Groq API calls (fast mode)
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            Dictionary mapping device problem -> IMDRF code
        """
        results = {}

        # Step 1: Collect unique problems
        unique_problems = set()
        for prob in device_problems:
            if self._is_blank(prob):
                continue
            raw = str(prob).strip()
            if raw and raw.lower() != "nan":
                # Handle semicolon-separated problems
                for part in raw.split(";"):
                    part = part.strip()
                    if part and part.lower() != "nan":
                        unique_problems.add(part)

        total_unique = len(unique_problems)
        if progress_callback:
            progress_callback(0, total_unique, f"Processing {total_unique} unique device problems...")

        # Step 2: Deterministic mapping pass (fast)
        needs_groq = []
        processed = 0

        for prob in unique_problems:
            pn = self._norm_term(prob)

            # Check cache first
            if pn in self.cache:
                results[prob] = self.cache[pn]
                processed += 1
                continue

            # Try deterministic match
            code = self._map_one_problem_to_code(prob)
            if code:
                results[prob] = code
                self.cache[pn] = code
                processed += 1
            else:
                needs_groq.append(prob)

            if progress_callback and processed % 100 == 0:
                progress_callback(processed, total_unique, f"Deterministic mapping: {processed}/{total_unique}")

        # Save cache after deterministic pass
        self._save_cache()

        if progress_callback:
            progress_callback(processed, total_unique, f"Deterministic mapping complete. {len(needs_groq)} problems need AI mapping.")

        # Step 3: Groq batch processing (if enabled and needed)
        if not deterministic_only and needs_groq and self.groq_client and self.groq_client.available:
            groq_results = self._batch_groq_mapping(needs_groq, progress_callback, processed, total_unique)
            results.update(groq_results)

            # Cache Groq results
            for prob, code in groq_results.items():
                pn = self._norm_term(prob)
                self.cache[pn] = code
            self._save_cache()
        elif needs_groq:
            # Mark as empty for problems that couldn't be mapped
            for prob in needs_groq:
                results[prob] = ""

        if progress_callback:
            progress_callback(total_unique, total_unique, "Mapping complete!")

        return results

    def _batch_groq_mapping(
        self,
        problems: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]],
        start_count: int,
        total_count: int
    ) -> Dict[str, str]:
        """
        Batch Groq API calls for multiple problems.

        Uses a single API call with multiple problems to reduce API overhead.
        Falls back to individual calls if batch fails.
        """
        results = {}

        # Process in batches
        for i in range(0, len(problems), GROQ_BATCH_SIZE):
            batch = problems[i:i + GROQ_BATCH_SIZE]
            batch_num = i // GROQ_BATCH_SIZE + 1
            total_batches = (len(problems) + GROQ_BATCH_SIZE - 1) // GROQ_BATCH_SIZE

            if progress_callback:
                progress_callback(
                    start_count + i,
                    total_count,
                    f"AI mapping batch {batch_num}/{total_batches}..."
                )

            # Try batch API call first
            batch_results = self._groq_batch_call(batch)

            if batch_results:
                results.update(batch_results)
            else:
                # Fallback to individual calls if batch fails
                for prob in batch:
                    code = self._groq_fallback(prob)
                    results[prob] = code if code else ""

            # Rate limiting delay between batches
            if i + GROQ_BATCH_SIZE < len(problems):
                time.sleep(GROQ_RATE_LIMIT_DELAY)

        return results

    def _groq_batch_call(self, problems: List[str]) -> Optional[Dict[str, str]]:
        """
        Make a single Groq API call for multiple problems.

        Returns None if the batch call fails, indicating fallback to individual calls.
        """
        if not self.groq_client or not self.groq_client.available:
            return None

        if not problems:
            return {}

        # Build batch prompt
        problems_list = "\n".join([f'{i+1}. "{p}"' for i, p in enumerate(problems)])
        terms_str = ', '.join([f'"{t}"' for t in self.level1_terms[:50]])

        prompt = f"""You are mapping device problems to IMDRF Level-1 codes.

Device Problems to map:
{problems_list}

Available Level-1 terms (from IMDRF Annex):
{terms_str}

CRITICAL RULES:
- Return ONLY valid JSON
- For each problem number, provide the most appropriate Level-1 term from the list
- If no good match exists, use "NO_MATCH"
- Return a JSON object with problem numbers as keys

Return format (JSON only):
{{"1": "<term or NO_MATCH>", "2": "<term or NO_MATCH>", ...}}"""

        try:
            response = self.groq_client.client.chat.completions.create(
                model=self.groq_client.model,
                messages=[
                    {"role": "system", "content": "You are an IMDRF coding expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Convert results back to problem -> code mapping
            batch_results = {}
            for i, prob in enumerate(problems):
                key = str(i + 1)
                selected = result.get(key, "NO_MATCH")

                if selected == "NO_MATCH" or not selected:
                    batch_results[prob] = ""
                else:
                    selected_norm = self._norm_term(selected)
                    if selected_norm in self.level1_map:
                        batch_results[prob] = self.level1_map[selected_norm]
                    else:
                        batch_results[prob] = ""

            return batch_results

        except Exception as e:
            print(f"Warning: Groq batch call failed: {e}")
            return None

    def process_dataframe_scalable(
        self,
        df: pd.DataFrame,
        device_problem_column: str,
        output_column: str = "IMDRF Code",
        deterministic_only: bool = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> pd.DataFrame:
        """
        Process an entire DataFrame with scalable IMDRF mapping.

        Automatically enables deterministic-only mode for large files.

        Args:
            df: Input DataFrame
            device_problem_column: Name of the column containing device problems
            output_column: Name of the output column for IMDRF codes
            deterministic_only: Force deterministic-only mode. If None, auto-detect based on file size.
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with IMDRF codes added
        """
        row_count = len(df)

        # Auto-detect mode based on file size
        if deterministic_only is None:
            deterministic_only = row_count >= LARGE_FILE_THRESHOLD
            if deterministic_only and progress_callback:
                progress_callback(0, row_count, f"Large file detected ({row_count} rows). Using fast deterministic-only mode.")

        # Extract all device problems
        device_problems = df[device_problem_column].tolist()

        # Batch process
        mapping = self.map_device_problems_batch(
            device_problems,
            deterministic_only=deterministic_only,
            progress_callback=progress_callback
        )

        # Apply mapping to DataFrame
        def get_code(device_problem):
            if self._is_blank(device_problem):
                return ""
            raw = str(device_problem).strip()
            if not raw or raw.lower() == "nan":
                return ""

            # Handle semicolon-separated problems
            parts = [p.strip() for p in raw.split(";") if p.strip()]
            if not parts:
                return ""

            codes_in_order = []
            seen = set()

            for part in parts:
                code = mapping.get(part, "")
                if not code:
                    continue
                if code not in seen:
                    seen.add(code)
                    codes_in_order.append(code)

            return " | ".join(codes_in_order) if codes_in_order else ""

        df[output_column] = df[device_problem_column].apply(get_code)

        return df

    def get_mapping_stats(self, mapping_results: Dict[str, str]) -> Dict:
        """
        Get statistics about mapping results.

        Returns:
            Dictionary with mapping statistics
        """
        total = len(mapping_results)
        mapped = sum(1 for code in mapping_results.values() if code)
        unmapped = total - mapped

        # Count by level
        level1_count = sum(1 for code in mapping_results.values() if code and len(code) == 3)
        level2_count = sum(1 for code in mapping_results.values() if code and len(code) == 5)
        level3_count = sum(1 for code in mapping_results.values() if code and len(code) == 7)

        return {
            "total_unique_problems": total,
            "mapped": mapped,
            "unmapped": unmapped,
            "mapping_rate": f"{(mapped/total*100):.1f}%" if total > 0 else "0%",
            "level1_codes": level1_count,
            "level2_codes": level2_count,
            "level3_codes": level3_count,
            "cache_size": len(self.cache)
        }
