"""
Manufacturer normalization with deterministic cleanup and web-verified canonicalization.
NON-NEGOTIABLE: Modify ONLY existing Manufacturer column in place. No new columns. No guessing.

Scalability Features:
- Deterministic-only mode for large datasets (skip web verification)
- Batch processing support
- Aggressive caching to avoid repeated lookups
"""
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Callable
from backend.groq_client import GroqClient
import requests
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup


# Threshold for deterministic-only mode (unique manufacturers)
LARGE_MANUFACTURER_THRESHOLD = 100


class ManufacturerNormalizer:
    """Normalizes manufacturer names deterministically with optional web-verified canonicalization."""
    
    # Strict domain allowlist (hard-coded, deterministic)
    PRIMARY_DOMAINS = [
        "sec.gov",           # EDGAR filings
        "sedarplus.ca",      # Canadian filings
        "gov.uk"             # Companies House
    ]
    
    SECONDARY_DOMAINS = [
        "asic.gov.au",       # Australia regulator
        "e-justice.europa.eu",  # EU business registers
        "opencorporates.com"  # Aggregator (must reference provenance)
    ]
    
    # Transaction keywords for verification
    TRANSACTION_KEYWORDS = [
        "acquired", "acquisition", "merger", "merged", "purchase",
        "will acquire", "has acquired"
    ]
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        try:
            self.groq_client = groq_client if groq_client is not None else GroqClient()
        except Exception as e:
            print(f"Warning: Could not initialize GroqClient: {e}. Manufacturer normalization will use deterministic methods only.")
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
        
        # Cache files
        self.canonical_map_file = os.path.join(self.cache_dir, "manufacturer_canonical_map.json")
        self.sources_map_file = os.path.join(self.cache_dir, "manufacturer_canonical_map_sources.json")
        self.nochange_map_file = os.path.join(self.cache_dir, "manufacturer_nochange.json")
        
        # Load caches
        self.canonical_map = self._load_json(self.canonical_map_file)
        self.sources_map = self._load_json(self.sources_map_file)
        self.nochange_map = self._load_json(self.nochange_map_file)
        
        # Legal suffixes (in order of removal)
        self.legal_suffixes = [
            "ltd", "llp", "inc", "corp", "company", "co", "gmbh", "ag", "sa", "sarl",
            "bv", "plc", "pvt", "limited"
        ]
        
        # Performance settings
        self.request_timeout = 10  # seconds
        self.max_retries = 2
        self.request_delay = 0.75  # seconds between requests
    
    def _load_json(self, filepath: str) -> dict:
        """Load JSON file, return empty dict on error."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_json(self, filepath: str, data: dict):
        """Save JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save {filepath}: {e}")
    
    def _deterministic_clean(self, name: str) -> str:
        """
        Deterministic cleanup of manufacturer name.
        - lowercase
        - trim
        - collapse multiple spaces
        - normalize punctuation at ends (strip trailing ".,;:()[]")
        - remove trailing legal suffix tokens repeatedly (loop until no suffix remains)
        - Must correctly handle "co.,", "co.", "co," as "co"
        """
        if not name or str(name).strip().lower() in ['nan', '', 'none', 'null']:
            return ""
        
        # lowercase, trim
        name = str(name).strip().lower()
        
        # collapse multiple spaces
        name = re.sub(r"\s+", " ", name)
        
        # Remove legal suffixes repeatedly until no suffix remains
        # Remove trailing punctuation BEFORE checking last token each iteration
        changed = True
        while changed:
            changed = False
            
            # Step 1: Remove trailing punctuation first (before checking last token)
            # Strip trailing ".,;:()[]"
            name = re.sub(r"[.,;:()\[\]]+$", "", name).strip()
            
            if not name:
                break
            
            # Step 2: Check if last token matches any legal suffix
            words = name.split()
            if not words:
                break
            
            last_token = words[-1].lower()
            # Remove any remaining punctuation from last token for comparison
            last_token_clean = re.sub(r"[.,;:()\[\]]+$", "", last_token)
            
            # Check against legal suffixes
            # Handle "co.,", "co.", "co," as "co"
            for suffix in self.legal_suffixes:
                if last_token_clean == suffix:
                    # Remove the last token (suffix)
                    words = words[:-1]
                    name = ' '.join(words).strip()
                    changed = True
                    break
        
        # Final cleanup: remove any remaining trailing punctuation
        name = re.sub(r"[.,;:()\[\]]+$", "", name).strip()
        
        return name.strip()
    
    def _groq_suggest_canonical(self, cleaned_name: str) -> Dict:
        """
        Get Groq suggestion with strict JSON contract.
        Output MUST be STRICT JSON:
        {
          "action": "NO_CHANGE" | "PROPOSE",
          "canonical": "<proposed canonical company name if PROPOSE else empty>",
          "relation": "acquired_by" | "merged_into" | "renamed_to" | "subsidiary_of" | "unknown",
          "search_queries": ["<query1>", "<query2>"]
        }
        """
        if not self.groq_client or not self.groq_client.available:
            return {"action": "NO_CHANGE", "canonical": "", "relation": "unknown", "search_queries": []}
        prompt = f"""You are a medical device industry expert. Given a manufacturer name, determine if it has been acquired, merged, renamed, or is a subsidiary of another company.

Manufacturer name: {cleaned_name}

CRITICAL RULES:
- Return ONLY valid JSON, no extra text
- Return action="NO_CHANGE" if uncertain (default to NO_CHANGE)
- Only return action="PROPOSE" if you are HIGHLY confident (90%+)
- If PROPOSE, provide canonical name, relation type, and search queries for verification

Return format (STRICT JSON only):
{{
  "action": "NO_CHANGE" or "PROPOSE",
  "canonical": "<proposed canonical name if PROPOSE, else empty string>",
  "relation": "acquired_by" or "merged_into" or "renamed_to" or "subsidiary_of" or "unknown",
  "search_queries": ["query1", "query2"]
}}"""

        try:
            response = self.groq_client.client.chat.completions.create(
                model=self.groq_client.model,
                messages=[
                    {"role": "system", "content": "You are a medical device industry expert. Return only valid JSON, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate Groq output
            if not isinstance(result, dict):
                return {"action": "NO_CHANGE", "canonical": "", "relation": "unknown", "search_queries": []}
            
            action = result.get("action", "NO_CHANGE")
            if action != "PROPOSE":
                return {"action": "NO_CHANGE", "canonical": "", "relation": "unknown", "search_queries": []}
            
            canonical = result.get("canonical", "").strip()
            if not canonical:
                return {"action": "NO_CHANGE", "canonical": "", "relation": "unknown", "search_queries": []}
            
            return {
                "action": "PROPOSE",
                "canonical": canonical,
                "relation": result.get("relation", "unknown"),
                "search_queries": result.get("search_queries", [])
            }
            
        except Exception as e:
            print(f"Warning: Groq manufacturer suggestion failed: {e}")
            return {"action": "NO_CHANGE", "canonical": "", "relation": "unknown", "search_queries": []}
    
    def _is_allowlisted_domain(self, url: str) -> bool:
        """Check if URL is from allowlisted domain (HARD STOP if not)."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check primary domains
            for primary in self.PRIMARY_DOMAINS:
                if domain == primary or domain.endswith('.' + primary):
                    return True
            
            # Check secondary domains
            for secondary in self.SECONDARY_DOMAINS:
                if domain == secondary or domain.endswith('.' + secondary):
                    return True
            
            # Check for official company domains (investor relations / press release pages)
            # Must be on company's own domain (basic heuristic: if domain contains company name tokens)
            # This is a simplified check - in production, you'd want a whitelist
            return False
            
        except:
            return False
    
    def _fetch_url(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Fetch URL content with retries and timeout.
        Returns (content_text, title) or None on failure.
        """
        if not self._is_allowlisted_domain(url):
            print(f"Warning: Non-allowlisted domain skipped: {url}")
            return None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    timeout=self.request_timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; MAUDE Processor)'},
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    # Parse HTML to extract text
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string if soup.title else ""
                    # Get text content (remove scripts, styles)
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    return (text, title)
                
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(self.request_delay)
                    continue
                print(f"Warning: Failed to fetch {url}: {e}")
        
        return None
    
    def _verify_relationship(self, cleaned_name: str, canonical_name: str, search_queries: List[str]) -> Tuple[bool, List[Dict]]:
        """
        Verify M&A relationship via allowlisted web sources.
        Returns (verified, sources_list)
        """
        all_domains = self.PRIMARY_DOMAINS + self.SECONDARY_DOMAINS
        verified_sources = []
        primary_verified = False
        secondary_count = 0
        
        # Normalize names for token matching
        cleaned_tokens = set(re.findall(r'\b\w+\b', cleaned_name.lower()))
        canonical_tokens = set(re.findall(r'\b\w+\b', canonical_name.lower()))
        
        # Try each search query
        for query in search_queries[:3]:  # Limit to 3 queries
            # For each allowlisted domain, construct search URL
            # Note: This is a simplified approach - in production, use proper search APIs
            for domain in all_domains:
                # Construct domain-specific search URL
                # This is a placeholder - actual implementation would use domain-specific search APIs
                # For now, we'll simulate by checking if we can construct valid URLs
                
                # Example: SEC EDGAR search
                if domain == "sec.gov":
                    # SEC EDGAR search URL pattern
                    search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={quote(query)}"
                elif domain == "sedarplus.ca":
                    search_url = f"https://www.sedarplus.ca/search/search_form_pc_en.jsf?lang=en&searchText={quote(query)}"
                elif domain == "gov.uk":
                    search_url = f"https://find-and-update.company-information.service.gov.uk/search?q={quote(query)}"
                elif domain == "opencorporates.com":
                    search_url = f"https://opencorporates.com/companies?q={quote(query)}"
                else:
                    # Skip domains without known search patterns
                    continue
                
                # Fetch and verify
                result = self._fetch_url(search_url)
                if result:
                    content, title = result
                    content_lower = content.lower()
                    
                    # Check for both names and transaction keyword
                    has_cleaned = any(token in content_lower for token in cleaned_tokens if len(token) > 3)
                    has_canonical = any(token in content_lower for token in canonical_tokens if len(token) > 3)
                    has_keyword = any(keyword in content_lower for keyword in self.TRANSACTION_KEYWORDS)
                    
                    if has_cleaned and has_canonical and has_keyword:
                        source_info = {
                            "url": search_url,
                            "domain": domain,
                            "title": title[:200] if title else "",
                            "retrieved_at": datetime.now().strftime("%Y-%m-%d")
                        }
                        verified_sources.append(source_info)
                        
                        if domain in self.PRIMARY_DOMAINS:
                            primary_verified = True
                        else:
                            secondary_count += 1
                        
                        # Rate limiting
                        time.sleep(self.request_delay)
        
        # Verification rules
        verified = primary_verified or (secondary_count >= 2)
        
        return verified, verified_sources
    
    def normalize(self, manufacturer_name: str) -> str:
        """
        Normalize manufacturer name.
        Returns cleaned name (deterministic) or canonical name (if verified M&A).
        HARD STOP: Never returns blank if input was non-blank.
        """
        if not manufacturer_name or str(manufacturer_name).strip().lower() in ['nan', '', 'none', 'null']:
            return ""
        
        # Step 1: Deterministic cleanup (always applied)
        cleaned = self._deterministic_clean(manufacturer_name)
        if not cleaned:
            return ""
        
        # Step 2: Canonicalization layer (frozen map lookup)
        if cleaned in self.canonical_map:
            canonical = self.canonical_map[cleaned]
            # HARD STOP: canonical must not be blank
            if not canonical or not canonical.strip():
                print(f"Warning: Canonical map has blank value for '{cleaned}', using cleaned name")
                return cleaned
            return canonical
        
        # Step 3: Check nochange cache
        if cleaned in self.nochange_map:
            # Already determined no change needed
            self.canonical_map[cleaned] = cleaned
            return cleaned
        
        # Step 4: Groq suggestion (semantic only, not trusted)
        groq_result = self._groq_suggest_canonical(cleaned)
        
        if groq_result.get("action") != "PROPOSE":
            # NO_CHANGE - cache it
            self.nochange_map[cleaned] = True
            self.canonical_map[cleaned] = cleaned
            self._save_json(self.nochange_map_file, self.nochange_map)
            self._save_json(self.canonical_map_file, self.canonical_map)
            return cleaned
        
        canonical_proposed = groq_result.get("canonical", "").strip()
        if not canonical_proposed:
            # Empty canonical - treat as NO_CHANGE
            self.nochange_map[cleaned] = True
            self.canonical_map[cleaned] = cleaned
            self._save_json(self.nochange_map_file, self.nochange_map)
            self._save_json(self.canonical_map_file, self.canonical_map)
            return cleaned
        
        # HARD STOP: canonical must not be blank
        if not canonical_proposed or not canonical_proposed.strip():
            print(f"Warning: Groq proposed blank canonical for '{cleaned}', using cleaned name")
            self.nochange_map[cleaned] = True
            self.canonical_map[cleaned] = cleaned
            return cleaned
        
        # Step 5: Web verification (allowlisted sources only)
        search_queries = groq_result.get("search_queries", [])
        if not search_queries:
            # No search queries - treat as NO_CHANGE
            self.nochange_map[cleaned] = True
            self.canonical_map[cleaned] = cleaned
            self._save_json(self.nochange_map_file, self.nochange_map)
            self._save_json(self.canonical_map_file, self.canonical_map)
            return cleaned
        
        verified, sources = self._verify_relationship(cleaned, canonical_proposed, search_queries)

        if verified and sources:
            # Update canonical map and sources map
            self.canonical_map[cleaned] = canonical_proposed
            self.sources_map[cleaned] = {
                "canonical": canonical_proposed,
                "sources": sources
            }
            self._save_json(self.canonical_map_file, self.canonical_map)
            self._save_json(self.sources_map_file, self.sources_map)
            return canonical_proposed
        else:
            # Verification failed - use cleaned original
            self.nochange_map[cleaned] = True
            self.canonical_map[cleaned] = cleaned
            self._save_json(self.nochange_map_file, self.nochange_map)
            self._save_json(self.canonical_map_file, self.canonical_map)
            return cleaned

    def normalize_deterministic_only(self, manufacturer_name: str) -> str:
        """
        Fast deterministic-only normalization (no web lookups, no Groq calls).
        Used for large datasets where web verification would be too slow.

        Args:
            manufacturer_name: Raw manufacturer name

        Returns:
            Cleaned manufacturer name (deterministic cleanup only)
        """
        if not manufacturer_name or str(manufacturer_name).strip().lower() in ['nan', '', 'none', 'null']:
            return ""

        # Step 1: Deterministic cleanup (always applied)
        cleaned = self._deterministic_clean(manufacturer_name)
        if not cleaned:
            return ""

        # Step 2: Check canonical map (use cached results if available)
        if cleaned in self.canonical_map:
            canonical = self.canonical_map[cleaned]
            if canonical and canonical.strip():
                return canonical

        # Step 3: Return deterministic cleaned name (skip Groq and web verification)
        return cleaned

    def normalize_batch(
        self,
        manufacturer_names: List[str],
        deterministic_only: bool = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, str]:
        """
        Batch normalize multiple manufacturer names efficiently.

        Automatically uses deterministic-only mode for large datasets.

        Args:
            manufacturer_names: List of manufacturer names to normalize
            deterministic_only: Force deterministic-only mode. If None, auto-detect.
            progress_callback: Optional callback(current, total, message) for progress

        Returns:
            Dictionary mapping original name -> normalized name
        """
        results = {}

        # Get unique names
        unique_names = set()
        for name in manufacturer_names:
            if name and str(name).strip().lower() not in ['nan', '', 'none', 'null']:
                unique_names.add(str(name).strip())

        total_unique = len(unique_names)

        # Auto-detect mode based on dataset size
        if deterministic_only is None:
            deterministic_only = total_unique >= LARGE_MANUFACTURER_THRESHOLD
            if deterministic_only and progress_callback:
                progress_callback(0, total_unique,
                    f"Large dataset ({total_unique} unique names). Using FAST deterministic-only mode.")

        if progress_callback:
            progress_callback(0, total_unique, f"Normalizing {total_unique} unique manufacturer names...")

        processed = 0
        for name in unique_names:
            if deterministic_only:
                results[name] = self.normalize_deterministic_only(name)
            else:
                results[name] = self.normalize(name)

            processed += 1
            if progress_callback and (processed % 500 == 0 or processed == total_unique):
                progress_callback(processed, total_unique, f"Normalized {processed}/{total_unique} names")

        # Save cache after batch processing
        self._save_json(self.canonical_map_file, self.canonical_map)
        self._save_json(self.nochange_map_file, self.nochange_map)

        if progress_callback:
            progress_callback(total_unique, total_unique, "Manufacturer normalization complete!")

        return results
