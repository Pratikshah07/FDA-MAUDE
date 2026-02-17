"""
AI-assisted column identification for MAUDE files.
Uses deterministic heuristics first, then Groq for semantic matching.
"""
import json
import hashlib
import os
from typing import Dict, Optional, List
from backend.groq_client import GroqClient


class ColumnIdentifier:
    """Identifies MAUDE columns using deterministic heuristics and Groq fallback."""
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        try:
            self.groq_client = groq_client if groq_client is not None else GroqClient()
        except Exception as e:
            print(f"Warning: Could not initialize GroqClient: {e}. Column identification will use deterministic methods only.")
            self.groq_client = None
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
        self.cache_file = os.path.join(self.cache_dir, "column_map_cache.json")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load column mapping cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save column mapping cache."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save column cache: {e}")
    
    def _normalize_header(self, header: str) -> str:
        """Normalize header for matching."""
        return ' '.join(str(header).strip().lower().split())
    
    def _deterministic_identify(self, headers: List[str]) -> Dict[str, Optional[str]]:
        """Use deterministic heuristics to identify columns."""
        result = {
            "event_date": None,
            "date_received": None,
            "manufacturer": None,
            "device_problem": None,
            "event_text": None
        }
        
        normalized_headers = {self._normalize_header(h): h for h in headers}
        
        # Event Date: contains "event" and "date"
        for norm, orig in normalized_headers.items():
            if "event" in norm and "date" in norm and result["event_date"] is None:
                result["event_date"] = orig
        
        # Date Received: contains "received" and "date"
        for norm, orig in normalized_headers.items():
            if "received" in norm and "date" in norm and result["date_received"] is None:
                result["date_received"] = orig
        
        # Manufacturer: equals/contains "manufacturer"
        for norm, orig in normalized_headers.items():
            if "manufacturer" in norm and result["manufacturer"] is None:
                result["manufacturer"] = orig
        
        # Device Problem: contains "device" and "problem"
        for norm, orig in normalized_headers.items():
            if "device" in norm and "problem" in norm and result["device_problem"] is None:
                result["device_problem"] = orig
        
        # Event Text: contains "event" and ("text" or "narrative")
        for norm, orig in normalized_headers.items():
            if "event" in norm and ("text" in norm or "narrative" in norm) and result["event_text"] is None:
                result["event_text"] = orig
        
        return result
    
    def _groq_identify(self, headers: List[str]) -> Dict[str, Optional[str]]:
        """Use Groq to identify columns semantically."""
        if not self.groq_client or not self.groq_client.available:
            return {
                "event_date": None,
                "date_received": None,
                "manufacturer": None,
                "device_problem": None,
                "event_text": None
            }
        
        headers_str = ', '.join([f'"{h}"' for h in headers])
        
        prompt = f"""You are identifying columns in a MAUDE (FDA medical device adverse event) data file.

Available column headers (exact strings, in order):
{headers_str}

You must identify which column corresponds to each of these required fields:
1. event_date: Column containing the event date
2. date_received: Column containing the date the report was received
3. manufacturer: Column containing manufacturer name
4. device_problem: Column containing device problem description
5. event_text: Column containing event narrative/text

CRITICAL RULES:
- Return ONLY a valid JSON object
- Use the EXACT column header string from the list above, or null if not found
- Do NOT rename or modify column names
- Do NOT guess - if uncertain, use null

Return format (JSON only, no explanations):
{{
  "event_date": "<exact header or null>",
  "date_received": "<exact header or null>",
  "manufacturer": "<exact header or null>",
  "device_problem": "<exact header or null>",
  "event_text": "<exact header or null>"
}}"""

        try:
            response = self.groq_client.client.chat.completions.create(
                model=self.groq_client.model,
                messages=[
                    {"role": "system", "content": "You are a data schema expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            # Validate: all values must be in headers list or null
            validated = {}
            for key in ["event_date", "date_received", "manufacturer", "device_problem", "event_text"]:
                value = result_json.get(key)
                if value and value in headers:
                    validated[key] = value
                else:
                    validated[key] = None
            
            return validated
            
        except Exception as e:
            print(f"Warning: Groq column identification failed: {e}")
            return {
                "event_date": None,
                "date_received": None,
                "manufacturer": None,
                "device_problem": None,
                "event_text": None
            }
    
    def identify_columns(self, headers: List[str]) -> Dict[str, str]:
        """
        Identify MAUDE columns using deterministic heuristics and Groq fallback.
        
        Returns:
            Dictionary mapping field names to exact column headers
        """
        # Create cache key from header list
        cache_key = hashlib.md5('|'.join(sorted(headers)).encode()).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            print(f"Using cached column mapping for schema signature: {cache_key[:8]}...")
            return self.cache[cache_key]
        
        # Step 1: Deterministic identification
        result = self._deterministic_identify(headers)
        
        # Step 2: Groq fallback for missing columns (only if Groq is available)
        missing = [k for k, v in result.items() if v is None]
        if missing and self.groq_client and self.groq_client.available:
            print(f"Using Groq to identify missing columns: {missing}")
            groq_result = self._groq_identify(headers)
            
            # Merge results (prefer deterministic, use Groq for missing)
            for key in missing:
                if groq_result.get(key):
                    result[key] = groq_result[key]
        elif missing:
            print(f"Warning: Missing columns {missing} could not be identified (Groq unavailable). Using deterministic results only.")
        
        # HARD STOP: device_problem is required
        if not result["device_problem"]:
            raise RuntimeError("HARD STOP: device_problem column could not be identified. Cannot place IMDRF Code adjacent.")
        
        # Save to cache
        self.cache[cache_key] = result
        self._save_cache()
        
        return result
