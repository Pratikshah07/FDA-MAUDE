"""
Groq API client for semantic tasks only.
Used for:
1. Manufacturer merger resolution
2. Event text keyword extraction
3. IMDRF semantic mapping
"""
import os
from config import GROQ_API_KEY, GROQ_MODEL


class GroqClient:
    """Client for interacting with Groq API."""

    def __init__(self):
        if not GROQ_API_KEY:
            # Graceful degradation: allow initialization without API key
            self.client = None
            self.model = GROQ_MODEL
            self.available = False
            print("Warning: GROQ_API_KEY not set. AI-assisted features will be disabled.")
        else:
            try:
                from groq import Groq  # lazy import — groq SDK is heavy; only load when actually needed
                self.client = Groq(api_key=GROQ_API_KEY)
                self.model = GROQ_MODEL
                self.available = True
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}. AI-assisted features will be disabled.")
                self.client = None
                self.available = False
    
    def resolve_manufacturer_merger(self, manufacturer_name: str) -> str:
        """
        Resolve manufacturer name considering mergers and acquisitions.
        Returns the resolved name only if confidence is HIGH.
        Otherwise returns the original cleaned name.
        
        Args:
            manufacturer_name: Cleaned manufacturer name
            
        Returns:
            Resolved manufacturer name or original if uncertain
        """
        if not self.available or not self.client:
            return manufacturer_name
        if not manufacturer_name or not manufacturer_name.strip():
            return manufacturer_name
        
        prompt = f"""You are a medical device industry expert. Given a manufacturer name, determine if it has been acquired by or merged with another company.

Manufacturer name: {manufacturer_name}

Rules:
- Only return a different name if you are HIGHLY confident (90%+) about a merger/acquisition
- If uncertain, return the EXACT input name unchanged
- Return ONLY the current/merged company name, nothing else
- Do not add explanations, confidence scores, or additional text
- If the company is independent or you're uncertain, return the input name as-is

Return format: Just the company name, or the input name if uncertain."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise medical device industry expert. Return only company names, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for deterministic responses
                max_tokens=50
            )
            
            resolved = response.choices[0].message.content.strip()
            
            # If response is too different or contains explanations, return original
            if len(resolved) > len(manufacturer_name) * 2:
                return manufacturer_name
            
            # If response is the same or very similar, return original
            if resolved.lower() == manufacturer_name.lower():
                return manufacturer_name
            
            # Only return resolved if it's clearly different and reasonable
            return resolved
            
        except Exception as e:
            # On any error, return original
            print(f"Groq API error in manufacturer resolution: {e}")
            return manufacturer_name
    
    def extract_keywords(self, event_text: str) -> str:
        """
        Extract medically and technically meaningful safety keywords from event text.
        
        Args:
            event_text: Raw event text from MAUDE report
            
        Returns:
            Comma-separated keywords in double quotes, or blank if nothing meaningful
        """
        if not self.available or not self.client:
            return ""
        if not event_text or not event_text.strip():
            return ""
        
        prompt = f"""Extract medically and technically meaningful safety-related keywords from this medical device adverse event text.

Event text: {event_text[:2000]}

CRITICAL RULES:
- Extract ONLY keywords that are medically or technically significant for device safety
- Focus on: malfunctions, failures, defects, injuries, complications, technical issues
- Each keyword must be enclosed in double quotes
- Multiple keywords should be comma-separated
- Examples of good keywords: "device malfunction", "battery failure", "overheating", "patient injury", "detachment", "breakage", "sparking", "failure to deliver energy"
- Do NOT include generic words like "device", "patient", "reported", "event", "description"
- Do NOT include the phrase "Event Description" or similar
- If nothing meaningful is found, return exactly: BLANK
- Return format: "keyword1", "keyword2", "keyword3"
- Return ONLY the quoted keywords, no explanations, no additional text

Extract keywords now:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical device safety analyst. Extract only meaningful safety keywords."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            keywords = response.choices[0].message.content.strip()
            
            # Validate format - should contain quoted keywords
            if not keywords or '"' not in keywords:
                return ""
            
            # Clean up any extra text
            if keywords.startswith('"') or keywords.startswith("Keywords:"):
                # Extract just the quoted keywords
                import re
                matches = re.findall(r'"([^"]+)"', keywords)
                if matches:
                    return ', '.join(f'"{m}"' for m in matches)
            
            return keywords
            
        except Exception as e:
            print(f"Groq API error in keyword extraction: {e}")
            return ""
    
    def map_to_imdrf(self, device_problem: str, imdrf_structure: dict) -> str:
        """
        Map device problem to IMDRF code using STRICT hierarchical matching.
        Codes are ONLY selected from the annexure file - NEVER generated.
        
        Args:
            device_problem: Device problem description
            imdrf_structure: Structured IMDRF Annexure A-G data (hierarchical)
            
        Returns:
            IMDRF code from deepest confident level, or blank if no match
        """
        if not self.available or not self.client:
            return ""
        if not device_problem or not device_problem.strip():
            return ""
        
        if not imdrf_structure or not any(imdrf_structure.values()):
            # No structure = no codes (strict rule)
            return ""
        
        # Build hierarchical structure description for the prompt
        structure_desc = self._build_hierarchical_imdrf_description(imdrf_structure)
        
        # Get all valid codes for validation
        all_codes = []
        for level in ['Level-1', 'Level-2', 'Level-3']:
            if level in imdrf_structure:
                codes_in_level = list(imdrf_structure[level].keys())
                all_codes.extend(codes_in_level)
        
        if not all_codes:
            print(f"ERROR: No IMDRF codes found in structure. Structure keys: {list(imdrf_structure.keys())}")
            for level in ['Level-1', 'Level-2', 'Level-3']:
                if level in imdrf_structure:
                    print(f"  {level} has {len(imdrf_structure[level])} entries but no valid codes")
            return ""
        
        # Debug: Show sample of codes
        if len(all_codes) > 0:
            print(f"  Sample codes available: {all_codes[:5]}... (total: {len(all_codes)})")
        
        system_prompt = """You are an expert medical device regulatory coding specialist with deep expertise in IMDRF (International Medical Device Regulators Forum) terminology and coding systems.

YOUR EXPERTISE:
- You understand medical device problems, malfunctions, and adverse events
- You are familiar with IMDRF Annex A-G coding structures
- You can match device problem descriptions to appropriate IMDRF codes
- You understand hierarchical coding (Level-1 = broad, Level-2 = specific, Level-3 = most specific)

CRITICAL MAPPING RULES:
1. You MUST ONLY select codes that exist in the provided IMDRF structure - NEVER generate codes
2. Use intelligent semantic matching - understand the MEANING of the device problem, not just keywords
3. Hierarchical matching strategy:
   a. First, identify the general category at Level-1 (e.g., "Device Malfunction", "Material Problem")
   b. Then, if available, find a more specific Level-2 code (e.g., "Mechanical Problem", "Electrical Problem")
   c. Finally, if available, find the most specific Level-3 code (e.g., "Battery Failure", "Circuit Failure")
   d. Always select the DEEPEST level where you have HIGH confidence
4. Semantic understanding:
   - "Difficult to Open or Close" → Look for codes related to mechanical operation, opening/closing mechanisms
   - "Detachment" → Look for codes related to component detachment, separation
   - "Intermittent Energy Output" → Look for codes related to energy delivery, power issues
   - "Device Malfunction" → Look for general malfunction codes
5. Return ONLY the exact code as it appears in the structure (preserve format exactly - could be "A0101", "A-01-01", etc.)
6. If you find a clear match at ANY level with confidence ≥80%, return that code
7. If uncertain at ALL levels (confidence <80%), return exactly: BLANK
8. Do NOT add explanations, descriptions, or any other text - ONLY the code or BLANK"""

        # Build a more focused prompt with relevant examples
        sample_codes = sorted(set(all_codes))[:30]  # Show first 30 codes
        
        user_prompt = f"""Map this device problem to an IMDRF code using intelligent hierarchical semantic matching.

DEVICE PROBLEM TO MAP:
"{device_problem[:600]}"

IMDRF STRUCTURE (hierarchical - search through this):
{structure_desc[:2500]}

AVAILABLE CODES (you MUST select from these codes in the structure above):
{', '.join(sample_codes)}
{'... and ' + str(len(all_codes) - 30) + ' more codes available in structure' if len(all_codes) > 30 else ''}

INTELLIGENT MAPPING PROCESS:
Step 1: Understand the device problem semantically
- What is the core issue? (malfunction, detachment, energy problem, material issue, etc.)
- What type of problem is it? (mechanical, electrical, software, material, etc.)

Step 2: Hierarchical search
- Level-1: Find the broadest category that matches (e.g., "Device Malfunction", "Material Problem")
- Level-2: If Level-1 match found, search for more specific subcategory
- Level-3: If Level-2 match found, search for most specific code

Step 3: Select deepest confident match
- Choose the DEEPEST level where you have ≥80% confidence
- If multiple levels match, prefer the most specific (Level-3 > Level-2 > Level-1)

SEMANTIC MATCHING EXAMPLES:
- "Difficult to Open or Close" → Look for: mechanical operation, opening/closing, access problems
- "Detachment of Device or Device Component" → Look for: detachment, separation, component issues
- "Intermittent Energy Output" → Look for: energy delivery, power supply, intermittent issues
- "Device Malfunction" → Look for: general malfunction, device failure codes
- "Battery Failure" → Look for: battery, power source, energy storage issues

OUTPUT FORMAT:
- Return ONLY the code exactly as it appears in the structure (e.g., "A0101", "A-01-01", "B0203")
- If you find a confident match (≥80%), return that code
- If uncertain (<80% confidence), return exactly: BLANK
- Do NOT add any explanations, descriptions, or additional text

Return ONLY the IMDRF code or BLANK:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Slightly higher for better semantic understanding
                max_tokens=50,  # Increased to allow for longer code formats
                top_p=0.9  # Nucleus sampling for better quality
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean and validate the response
            code_clean = code.strip()
            
            # Check if blank
            if not code or code_clean.upper() in ['BLANK', 'NONE', 'N/A', 'NA', '', 'NO MATCH', 'NO CODE', 'NONE FOUND']:
                return ""
            
            # Try to find matching code in structure (flexible matching)
            code_found = None
            
            # First try exact match (case-insensitive)
            for level in ['Level-1', 'Level-2', 'Level-3']:
                if level in imdrf_structure:
                    for existing_code in imdrf_structure[level].keys():
                        if existing_code.strip().upper() == code_clean.upper():
                            code_found = existing_code  # Use original format
                            break
                    if code_found:
                        break
            
            # If not found, try normalized matching (remove hyphens, spaces)
            if not code_found:
                code_normalized = code_clean.upper().replace('-', '').replace('_', '').replace(' ', '')
                for level in ['Level-1', 'Level-2', 'Level-3']:
                    if level in imdrf_structure:
                        for existing_code in imdrf_structure[level].keys():
                            existing_normalized = existing_code.upper().replace('-', '').replace('_', '').replace(' ', '')
                            if existing_normalized == code_normalized:
                                code_found = existing_code  # Use original format
                                break
                        if code_found:
                            break
            
            # If still not found, try partial match (code contains or is contained)
            if not code_found and len(code_clean) >= 3:
                for level in ['Level-1', 'Level-2', 'Level-3']:
                    if level in imdrf_structure:
                        for existing_code in imdrf_structure[level].keys():
                            if code_clean.upper() in existing_code.upper() or existing_code.upper() in code_clean.upper():
                                code_found = existing_code  # Use original format
                                break
                        if code_found:
                            break
            
            if code_found:
                return code_found  # Return in original format from structure
            
            # Code not found in structure - try one more time with more flexible matching
            # Sometimes codes come with extra text like "Code: A0101" or "IMDRF: A0101"
            import re
            code_extracted = re.search(r'([A-G][\s\-]?[\d\s\-]{3,6})', code_clean.upper())
            if code_extracted:
                potential_code = code_extracted.group(1).replace(' ', '').replace('-', '')
                # Try to find a code that matches when normalized
                for level in ['Level-1', 'Level-2', 'Level-3']:
                    if level in imdrf_structure:
                        for existing_code in imdrf_structure[level].keys():
                            existing_normalized = existing_code.upper().replace('-', '').replace('_', '').replace(' ', '')
                            if existing_normalized == potential_code or potential_code in existing_normalized or existing_normalized in potential_code:
                                return existing_code  # Return in original format
            
            # Code not found in structure - return blank (strict rule)
            # Log for debugging (only for first few to avoid spam)
            if len(code_clean) > 0 and code_clean.upper() not in ['BLANK', 'NONE'] and len(code_clean) < 20:
                # Only log short codes to avoid spam from long error messages
                pass  # Suppress logging to reduce noise
            return ""
            
        except Exception as e:
            print(f"Groq API error in IMDRF mapping: {e}")
            return ""
    
    def _build_hierarchical_imdrf_description(self, imdrf_structure: dict) -> str:
        """Build a hierarchical text description of IMDRF structure for prompts."""
        if not imdrf_structure:
            return "IMDRF structure not available. Return BLANK."
        
        desc_parts = []
        
        # Build Level-1 (broad categories)
        if 'Level-1' in imdrf_structure and imdrf_structure['Level-1']:
            desc_parts.append("\n=== Level-1 (Broad Categories) ===")
            for code, info in list(imdrf_structure['Level-1'].items())[:20]:  # Limit to avoid token limits
                desc = info.get('description', '')
                annex = info.get('annex', '')
                desc_parts.append(f"  Code: {code} | Description: {desc} | Annex: {annex}")
            if len(imdrf_structure['Level-1']) > 20:
                desc_parts.append(f"  ... and {len(imdrf_structure['Level-1']) - 20} more Level-1 codes")
        
        # Build Level-2 (mid-level)
        if 'Level-2' in imdrf_structure and imdrf_structure['Level-2']:
            desc_parts.append("\n=== Level-2 (Mid-Level Categories) ===")
            for code, info in list(imdrf_structure['Level-2'].items())[:20]:
                desc = info.get('description', '')
                annex = info.get('annex', '')
                parent = info.get('parent', '')
                desc_parts.append(f"  Code: {code} | Description: {desc} | Annex: {annex} | Parent: {parent}")
            if len(imdrf_structure['Level-2']) > 20:
                desc_parts.append(f"  ... and {len(imdrf_structure['Level-2']) - 20} more Level-2 codes")
        
        # Build Level-3 (specific)
        if 'Level-3' in imdrf_structure and imdrf_structure['Level-3']:
            desc_parts.append("\n=== Level-3 (Specific Codes) ===")
            for code, info in list(imdrf_structure['Level-3'].items())[:20]:
                desc = info.get('description', '')
                annex = info.get('annex', '')
                parent = info.get('parent', '')
                desc_parts.append(f"  Code: {code} | Description: {desc} | Annex: {annex} | Parent: {parent}")
            if len(imdrf_structure['Level-3']) > 20:
                desc_parts.append(f"  ... and {len(imdrf_structure['Level-3']) - 20} more Level-3 codes")
        
        if not desc_parts:
            return "IMDRF structure is empty. Return BLANK."
        
        return "\n".join(desc_parts)
