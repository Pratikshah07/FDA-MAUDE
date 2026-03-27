"""
Static parent-company map for MAUDE manufacturer normalization.
Applied at IMDRF Insights display level only — raw CSV files are never modified.

Keys   : lowercase stripped manufacturer names as they appear in processed MAUDE files.
Values : Title Case canonical parent company name.

Abbott Vascular is intentionally kept SEPARATE from Abbott.
Only subsidiary / child entity names are listed here; standalone parent names are NOT
included (they pass through unchanged).

Dynamic matching (applied in apply_parent_map before static lookup):
  1. Pre-processing  — strips trailing numeric IDs, slash-separated alternates,
                       normalises intra-word hyphens and American spelling variants.
  2. Prefix matching — if the cleaned name *starts with* a well-known brand token
                       (≥ 5 chars, or a 2-word phrase) it is mapped to that parent.
                       Longer prefixes are tested first so "abbott vascular" wins
                       over "abbott".  Only conservative, industry-specific tokens
                       are used to minimise false-positive risk.
"""
import re as _re

# Raw map: child name (lowercase) → Title Case parent name
# Also includes base parent names (lowercase) → Title Case so all rows consolidate
# under a single key after the pipeline's lowercase normalization.
_RAW_MAP: dict = {

    # ── Base parent names (Title Case normalisation only) ──────────────────
    "medtronic":                                "Medtronic",
    "biotronik":                                "Biotronik",
    "medinol":                                  "Medinol",
    "boston scientific":                        "Boston Scientific",
    "becton dickinson":                         "Becton Dickinson",
    "abbott":                                   "Abbott",
    "abbott vascular":                          "Abbott Vascular",
    "johnson & johnson":                        "Johnson & Johnson",
    "philips":                                  "Philips",
    "stryker":                                  "Stryker",
    "zimmer biomet":                            "Zimmer Biomet",
    "edwards lifesciences":                     "Edwards Lifesciences",
    "teleflex":                                 "Teleflex",
    "smith & nephew":                           "Smith & Nephew",
    "baxter":                                   "Baxter",
    "hologic":                                  "Hologic",
    "integra lifesciences":                     "Integra Lifesciences",
    "icu medical":                              "ICU Medical",
    "ge healthcare":                            "GE Healthcare",

    # ── Medtronic subsidiaries ─────────────────────────────────────────────
    "medtronic ireland":                        "Medtronic",
    "medtronic mexico":                         "Medtronic",
    "medtronic vascular":                       "Medtronic",
    "medtronic cardiovascular santa rosa":      "Medtronic",
    "medtronic / medtronic ireland":            "Medtronic",
    "medtronic plc":                            "Medtronic",
    "medtronic usa":                            "Medtronic",
    "medtronic sofamor danek":                  "Medtronic",
    "medtronic sofamor danek usa":              "Medtronic",
    "medtronic neurological":                   "Medtronic",
    "medtronic minimed":                        "Medtronic",
    "medtronic diabetes":                       "Medtronic",
    "covidien":                                 "Medtronic",   # acquired 2015
    "covidien llp":                             "Medtronic",
    "covidien lp":                              "Medtronic",

    # ── Biotronik ──────────────────────────────────────────────────────────
    "biotronik ag buelach switzerland":         "Biotronik",
    "biotronik ag":                             "Biotronik",
    "biotronik se & co kg":                     "Biotronik",
    "biotronik se & co":                        "Biotronik",
    "biotronik se":                             "Biotronik",
    "biotronik inc":                            "Biotronik",

    # ── Medinol ────────────────────────────────────────────────────────────
    "medinol ltd":                              "Medinol",
    "medinol ltd (jerusalem":                   "Medinol",   # parenthesis cut-off variant
    "medinol ltd jerusalem":                    "Medinol",

    # ── Boston Scientific (corporation + facilities) ──────────────────────
    "boston scientific corporation":            "Boston Scientific",
    "boston scientific scimed":                 "Boston Scientific",
    "boston scientific scimed inc":             "Boston Scientific",
    "boston scientific neuromodulation":        "Boston Scientific",
    "boston scientific neuromodulation corp":   "Boston Scientific",
    "boston scientific - galway":               "Boston Scientific",
    "boston scientific - maple grove":          "Boston Scientific",
    "boston scientific - natick":               "Boston Scientific",
    "boston scientific - ireland":              "Boston Scientific",
    "boston scientific - endoscopy":            "Boston Scientific",
    "boston scientific - miami":                "Boston Scientific",
    "boston scientific - san jose":             "Boston Scientific",
    "boston scientific - fremont":              "Boston Scientific",

    # ── Becton Dickinson ───────────────────────────────────────────────────
    "becton dickinson and company":             "Becton Dickinson",
    "bd":                                       "Becton Dickinson",
    "bd medical":                               "Becton Dickinson",
    "carefusion":                               "Becton Dickinson",   # acquired 2015
    "carefusion corporation":                   "Becton Dickinson",
    "carefusion 303":                           "Becton Dickinson",
    "carefusion solutions":                     "Becton Dickinson",
    "c.r. bard":                                "Becton Dickinson",   # acquired 2017
    "cr bard":                                  "Becton Dickinson",
    "bard medical":                             "Becton Dickinson",
    "bard medical division":                    "Becton Dickinson",
    "bard access systems":                      "Becton Dickinson",

    # ── Abbott ─────────────────────────────────────────────────────────────
    # Abbott Vascular is kept as a SEPARATE entity (see section below)
    "abbott inc":                               "Abbott",
    "abbott laboratories":                      "Abbott",
    "st. jude medical":                         "Abbott",    # acquired 2017
    "st jude medical":                          "Abbott",
    "st. jude medical inc":                     "Abbott",
    "st jude medical inc":                      "Abbott",

    # ── Abbott Vascular (intentionally separate from Abbott) ───────────────
    "abbott vascular devices":                  "Abbott Vascular",
    "abbott vascular inc":                      "Abbott Vascular",

    # ── Johnson & Johnson ──────────────────────────────────────────────────
    "johnson and johnson":                      "Johnson & Johnson",
    "ethicon":                                  "Johnson & Johnson",
    "ethicon inc":                              "Johnson & Johnson",
    "ethicon endo-surgery":                     "Johnson & Johnson",
    "ethicon endosurgery":                      "Johnson & Johnson",
    "depuy":                                    "Johnson & Johnson",
    "depuy synthes":                            "Johnson & Johnson",
    "depuy synthes products":                   "Johnson & Johnson",
    "depuy spine":                              "Johnson & Johnson",
    "depuy orthopaedics":                       "Johnson & Johnson",
    "cordis":                                   "Johnson & Johnson",
    "cordis corporation":                       "Johnson & Johnson",
    "cordis cashel":                            "Johnson & Johnson",
    "cordis - cashel":                          "Johnson & Johnson",
    "biosense webster":                         "Johnson & Johnson",
    "biosense webster inc":                     "Johnson & Johnson",
    "acclarent":                                "Johnson & Johnson",

    # ── Philips ────────────────────────────────────────────────────────────
    "philips healthcare":                       "Philips",
    "philips medical systems":                  "Philips",
    "koninklijke philips":                      "Philips",
    "philips electronics":                      "Philips",

    # ── Siemens Healthineers ───────────────────────────────────────────────
    "siemens healthineers":                     "Siemens Healthineers",
    "siemens medical solutions":                "Siemens Healthineers",
    "siemens medical solutions usa":            "Siemens Healthineers",
    "siemens ag":                               "Siemens Healthineers",

    # ── GE Healthcare ──────────────────────────────────────────────────────
    "ge healthcare":                            "GE Healthcare",
    "ge medical systems":                       "GE Healthcare",
    "general electric company":                 "GE Healthcare",
    "gehealthcare":                             "GE Healthcare",

    # ── Stryker ────────────────────────────────────────────────────────────
    "stryker corporation":                      "Stryker",
    "stryker sales corporation":                "Stryker",
    "stryker orthopaedics":                     "Stryker",
    "stryker neurovascular":                    "Stryker",
    "stryker spine":                            "Stryker",
    "stryker endoscopy":                        "Stryker",
    "howmedica osteonics":                      "Stryker",

    # ── Zimmer Biomet ──────────────────────────────────────────────────────
    "zimmer":                                   "Zimmer Biomet",
    "zimmer inc":                               "Zimmer Biomet",
    "biomet":                                   "Zimmer Biomet",   # acquired 2015
    "biomet inc":                               "Zimmer Biomet",

    # ── Edwards Lifesciences ───────────────────────────────────────────────
    "edwards lifesciences llc":                 "Edwards Lifesciences",
    "edwards lifesciences corporation":         "Edwards Lifesciences",

    # ── Teleflex ───────────────────────────────────────────────────────────
    "teleflex medical":                         "Teleflex",
    "teleflex incorporated":                    "Teleflex",
    "arrow international":                      "Teleflex",   # acquired 2007
    "arrow international inc":                  "Teleflex",
    "lma international":                        "Teleflex",

    # ── Smith & Nephew ─────────────────────────────────────────────────────
    "smith and nephew":                         "Smith & Nephew",
    "smith & nephew inc":                       "Smith & Nephew",
    "smith & nephew orthopaedics":              "Smith & Nephew",

    # ── Baxter ─────────────────────────────────────────────────────────────
    "baxter healthcare":                        "Baxter",
    "baxter healthcare corporation":            "Baxter",
    "baxter international":                     "Baxter",

    # ── ICU Medical ────────────────────────────────────────────────────────
    "icu medical inc":                          "ICU Medical",
    "hospira":                                  "ICU Medical",

    # ── Hologic ────────────────────────────────────────────────────────────
    "hologic inc":                              "Hologic",

    # ── Integra Lifesciences ───────────────────────────────────────────────
    "integra lifesciences corporation":         "Integra Lifesciences",
}

# Normalised lookup dict — keys are already lowercase-stripped for fast O(1) lookup
PARENT_COMPANY_LOOKUP: dict = {k.strip().lower(): v for k, v in _RAW_MAP.items()}


# ── Pre-processing ────────────────────────────────────────────────────────────

def _preprocess_key(key: str) -> str:
    """
    Apply deterministic, reversible transforms to a lowercase manufacturer key
    before static-map lookup.  The original data is never touched.

    Transforms (in order):
      1. Strip trailing numeric IDs/phone numbers  e.g. " - 9616671"
      2. Strip slash-separated alternate names     e.g. " /howmedica osteonics"
      3. Replace intra-word hyphens with a space   e.g. "stryker-mahwah"
      4. American → British spelling for 'ortho*'  "orthopedics" → "orthopaedics"
         (the static map uses the British spelling throughout)
      5. Collapse multiple spaces and trim
    """
    # 1. Trailing numeric ID: " - 9616671" or "- 3015516266" (5+ digits)
    key = _re.sub(r'\s*[-\u2013]\s*\d{5,}\s*$', '', key)
    # 2. Slash-separated alternate company name
    key = _re.sub(r'\s*/.*$', '', key)
    # 3. Intra-word hyphen → space  ("stryker-mahwah" → "stryker mahwah")
    key = _re.sub(r'(?<=\w)-(?=\w)', ' ', key)
    # 4. American spelling normalisation
    key = key.replace('orthopedics', 'orthopaedics')
    # 5. Tidy up
    key = _re.sub(r'\s+', ' ', key).strip()
    return key


# ── Prefix-to-parent table ────────────────────────────────────────────────────
# Rules:
#   • Multi-word prefixes appear before single-word ones (more specific wins).
#   • Single-word prefixes must be ≥ 5 characters to avoid generic collisions.
#   • "abbott vascular" appears before "abbott" so it takes priority.
#   • A prefix matches only when followed by a space or at end-of-string,
#     preventing partial-word false positives (e.g. "strykerite" won't match "stryker").
_PREFIX_PARENT: list = sorted([
    # ── Multi-word / high-specificity ──────────────────────────────────────
    ("boston scientific",     "Boston Scientific"),
    ("becton dickinson",      "Becton Dickinson"),
    ("abbott vascular",       "Abbott Vascular"),    # must precede "abbott"
    ("edwards lifesciences",  "Edwards Lifesciences"),
    ("integra lifesciences",  "Integra Lifesciences"),
    ("smith & nephew",        "Smith & Nephew"),
    ("icu medical",           "ICU Medical"),
    ("ge healthcare",         "GE Healthcare"),
    ("siemens healthineers",  "Siemens Healthineers"),
    ("siemens medical",       "Siemens Healthineers"),
    ("johnson & johnson",     "Johnson & Johnson"),
    ("howmedica osteonics",   "Stryker"),             # standalone entity → Stryker
    ("depuy synthes",         "Johnson & Johnson"),   # must precede "depuy"
    ("depuy orthopaedics",    "Johnson & Johnson"),
    ("depuy spine",           "Johnson & Johnson"),
    # ── Single-word prefixes (≥ 5 chars) ───────────────────────────────────
    ("medtronic",             "Medtronic"),
    ("biotronik",             "Biotronik"),
    ("teleflex",              "Teleflex"),
    ("hologic",               "Hologic"),
    ("stryker",               "Stryker"),
    ("philips",               "Philips"),
    ("baxter",                "Baxter"),
    ("ethicon",               "Johnson & Johnson"),
    ("covidien",              "Medtronic"),
    ("carefusion",            "Becton Dickinson"),
    ("medinol",               "Medinol"),
    ("cordis",                "Johnson & Johnson"),
    ("zimmer",                "Zimmer Biomet"),
    ("biomet",                "Zimmer Biomet"),
    ("depuy",                 "Johnson & Johnson"),
    ("abbott",                "Abbott"),
    ("siemens",               "Siemens Healthineers"),
], key=lambda t: -len(t[0]))  # longest prefix first → most specific wins


def _prefix_match(key: str) -> str:
    """
    Return the parent company name if `key` starts with a known brand prefix,
    or an empty string if no prefix matches.

    The prefix must be the *entire* key or be followed by a space, so
    "stryker" matches "stryker orthopaedics mahwah" but NOT "strykerlite".
    """
    for prefix, parent in _PREFIX_PARENT:
        if key == prefix or key.startswith(prefix + ' '):
            return parent
    return ""


def apply_parent_map(df, mfr_col: str):
    """
    Apply the parent company map to the manufacturer column of a DataFrame.

    Non-destructive: only the in-memory copy is changed; the source file is untouched.

    Args:
        df      : pandas DataFrame containing the manufacturer column.
        mfr_col : Name of the manufacturer column to normalise.

    Returns:
        (df_copy, merge_log)

        df_copy   : Copy of df with manufacturer values replaced where a parent was found.
        merge_log : Sorted list of dicts recording every child→parent substitution that
                    actually occurred in this dataset:
                    [{"original": "medtronic ireland", "parent": "Medtronic"}, ...]
    """
    seen: dict = {}  # lowercase key → {original, parent}

    def _lookup(name: str) -> str:
        if not name or str(name).strip().lower() in ("", "nan", "none", "nat"):
            return name
        raw = str(name).strip()
        key = raw.lower()

        # ── Step 1: exact static-map match ───────────────────────────────────
        parent = PARENT_COMPANY_LOOKUP.get(key)
        if parent:
            if key not in seen:
                seen[key] = {"original": raw, "parent": parent}
            return parent

        # ── Step 2: pre-process then try exact match again ───────────────────
        processed = _preprocess_key(key)
        if processed and processed != key:
            parent = PARENT_COMPANY_LOOKUP.get(processed)
            if parent:
                if key not in seen:
                    seen[key] = {"original": raw, "parent": parent}
                return parent

        # ── Step 3: prefix match on the processed key ────────────────────────
        parent = _prefix_match(processed if processed else key)
        if parent:
            if key not in seen:
                seen[key] = {"original": raw, "parent": parent}
            return parent

        return raw

    df = df.copy()
    df[mfr_col] = df[mfr_col].apply(_lookup)
    merge_log = sorted(seen.values(), key=lambda x: x["original"])
    return df, merge_log
