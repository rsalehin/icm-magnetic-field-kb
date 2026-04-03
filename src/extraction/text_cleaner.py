# src/extraction/text_cleaner.py
"""
Page-level text cleaning for arXiv astrophysics PDFs.
"""
import re

# ── Strip patterns ────────────────────────────────────────────────────────────

ARXIV_STAMP = re.compile(r"arXiv:\S+\s+\d+\s+\w+\s+\d{4}\s*")

JOURNAL_HEADER = re.compile(
    r"^(Astronomy\s*&\s*Astrophysics|The Astrophysical Journal"
    r"|Monthly Notices|A&A|ApJ|MNRAS).*$",
    re.MULTILINE | re.IGNORECASE
)

# Running headers: "M. Murgia et al.: Title"
RUNNING_HEADER = re.compile(
    r"^[A-Z][^\n]{5,80}et al\.[^\n]{0,80}\n",
    re.MULTILINE
)

# Affiliation lines ending in country or institution keywords
AFFILIATION = re.compile(
    r"^\s*[\d\*†‡]?\s*[A-Z][^\n]{10,120}"
    r"(Italy|USA|Germany|France|UK|Spain|Netherlands|Australia"
    r"|Observatory|Institute|University|Universit|Department"
    r"|Dipartimento|INAF|CNRS|MPIfR|NRAO|ESO)\S*\s*$",
    re.MULTILINE
)

MANUSCRIPT_LINE = re.compile(
    r"^.*?(DOI:|manuscript no\.|Received|Accepted|Send offprint"
    r"|will be inserted|September \d+, \d{4}).*$",
    re.MULTILINE | re.IGNORECASE
)

STANDALONE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)

# Equation-only lines: "(6)" or isolated math
EQUATION_LABEL_LINE = re.compile(r"^\s*\(\d+\)\s*$", re.MULTILINE)

# Email lines
EMAIL_LINE = re.compile(r"^.*?@[^\s]+\.[a-z]{2,4}.*$", re.MULTILINE)

# "e-mail:", "correspondence to:" lines
CORRESPONDENCE_LINE = re.compile(
    r"^\s*(e-?mail|correspondence|contact).*$",
    re.MULTILINE | re.IGNORECASE
)


def clean_page(text: str) -> str:
    """Strip boilerplate from a single page's raw text."""
    text = ARXIV_STAMP.sub("", text)
    text = JOURNAL_HEADER.sub("", text)
    text = RUNNING_HEADER.sub("", text)
    text = EMAIL_LINE.sub("", text)
    text = CORRESPONDENCE_LINE.sub("", text)
    text = AFFILIATION.sub("", text)
    text = MANUSCRIPT_LINE.sub("", text)
    text = STANDALONE_NUMBER.sub("", text)
    text = EQUATION_LABEL_LINE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_soft_wrapped_lines(text: str) -> str:
    """Join soft-wrapped lines into full sentences."""
    lines  = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            result.append("")
            i += 1
            continue

        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            curr      = line.rstrip()

            # Hyphenated break
            if curr.endswith("-") and next_line and next_line[0].islower():
                result.append(curr[:-1])
                i += 1
                continue

            ends_sentence   = bool(re.search(r"[.!?]\s*$", curr))
            next_is_heading = bool(re.match(r"^\d+[\.\d]*\s+[A-Z]", next_line))
            next_is_empty   = not next_line

            if (not ends_sentence
                    and not next_is_heading
                    and not next_is_empty
                    and next_line
                    and not next_line[0].isupper()):
                result.append(curr + " ")
                i += 1
                continue

        result.append(line)
        i += 1

    return re.sub(r"  +", " ", "\n".join(result))


def split_into_paragraphs(text: str, min_chars: int = 150) -> list[str]:
    """
    Split cleaned text into paragraph-level chunks.
    min_chars raised to 150 to filter noise fragments.
    """
    text = join_soft_wrapped_lines(text)

    # Split on blank lines first
    blocks = re.split(r"\n\s*\n", text)

    sentence_split = re.compile(r"(?<=[.!?])\s{1,3}(?=[A-Z][a-z])")

    final_paras = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) > 400:
            sentences = sentence_split.split(block)
            current   = ""
            for sent in sentences:
                if len(current) + len(sent) < 350:
                    current = (current + " " + sent).strip()
                else:
                    if len(current) >= min_chars:
                        final_paras.append(re.sub(r"\s+", " ", current))
                    current = sent
            if len(current) >= min_chars:
                final_paras.append(re.sub(r"\s+", " ", current))
        else:
            clean = re.sub(r"\s+", " ", block)
            if len(clean) >= min_chars:
                final_paras.append(clean)

    return final_paras