#!/usr/bin/env python3
"""
pdf_date_rename.py - OCR PDFs and rename them with a YYYY-MM-DD prefix
for chronological sorting.

Usage:
    python3 pdf_date_rename.py [OPTIONS] [PDF_FILES_OR_DIRS...]

Options:
    -r, --recursive     Recursively scan directories for PDFs
    -n, --dry-run       Show renames without executing them
    -v, --verbose       Show detailed output including extracted text snippets
    --no-ocr            Skip OCR fallback; only use embedded PDF text
    --no-embedded       Skip embedded text; always use OCR
    --earliest          Use the earliest date found (default: first date found)
    --latest            Use the latest date found
    -h, --help          Show this help message

Examples:
    python3 pdf_date_rename.py invoice.pdf
    python3 pdf_date_rename.py -n -v ./documents/
    python3 pdf_date_rename.py -r --earliest ~/Downloads/
"""

import argparse
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional imports — give helpful messages if missing
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF is required. Install it with: pip install pymupdf")

try:
    from dateutil import parser as dateutil_parser
    from dateutil.parser import ParserError
except ImportError:
    sys.exit("python-dateutil is required. Install it with: pip install python-dateutil")

# pytesseract / Pillow are only needed when OCR is requested
def _import_ocr():
    try:
        import pytesseract
        from PIL import Image
        import io
        return pytesseract, Image, io
    except ImportError:
        sys.exit(
            "OCR requires pytesseract and Pillow.\n"
            "Install them with: pip install pytesseract Pillow\n"
            "You also need Tesseract installed: https://tesseract-ocr.github.io/"
        )


# ---------------------------------------------------------------------------
# Date extraction helpers
# ---------------------------------------------------------------------------

# Regex patterns ordered roughly from most- to least-specific.
# Each captures one potential date string to pass to dateutil.
_DATE_PATTERNS = [
    # ISO / sortable: 2023-01-15, 2023/01/15, 2023.01.15
    r"\b((?:19|20)\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2})\b",
    # US long form: January 15, 2023  /  Jan 15, 2023  /  Jan. 15, 2023
    r"\b((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+\d{1,2},?\s+(?:19|20)\d{2})\b",
    # Day-Month-Year long form: 15 January 2023  /  15 Jan 2023
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+(?:19|20)\d{2})\b",
    # Numeric US: 01/15/2023  or  1/15/2023
    r"\b(\d{1,2}/\d{1,2}/(?:19|20)\d{2})\b",
    # Numeric Euro: 15.01.2023
    r"\b(\d{1,2}\.\d{1,2}\.(?:19|20)\d{2})\b",
    # Month Year only: January 2023, Jan 2023
    r"\b((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+(?:19|20)\d{2})\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _DATE_PATTERNS]

# Reject clearly-bogus matches: page numbers, version strings, etc.
_REJECT_CONTEXT = re.compile(
    r"(?:version|ver|v|rev|page|pg|p\.)\s*\d",
    re.IGNORECASE,
)

# Already has a date prefix YYYY-MM-DD
_ALREADY_DATED = re.compile(r"^\d{4}-\d{2}-\d{2}[_\-\s]")


def _candidate_dates_from_text(text: str) -> list[datetime]:
    """Return a list of datetime objects parsed from date-like strings in text."""
    found = []
    seen_strings = set()

    for pattern in _COMPILED_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(1).strip()
            if raw in seen_strings:
                continue
            seen_strings.add(raw)

            # Skip if surrounding context looks like a version/page number
            start = max(0, m.start() - 10)
            context = text[start : m.start()]
            if _REJECT_CONTEXT.search(context):
                continue

            try:
                dt = dateutil_parser.parse(raw, default=datetime(1900, 1, 1))
                # Ignore dates with default year (i.e., year not found in raw string)
                if dt.year == 1900 and "1900" not in raw:
                    # Month-only patterns land here; set day to 1 but keep year
                    # Actually for month-year patterns dateutil gets year right
                    pass
                # Sanity check: year must be plausible
                current_year = datetime.now().year
                if not (1900 <= dt.year <= current_year + 1):
                    continue
                found.append(dt)
            except (ParserError, ValueError, OverflowError):
                continue

    return found


def extract_dates_from_text(text: str, strategy: str = "first") -> date | None:
    """
    Extract the best date from PDF text.

    strategy: "first"   – return the date that appears first in the text
              "earliest" – return the chronologically earliest date
              "latest"   – return the chronologically latest date
    """
    candidates = _candidate_dates_from_text(text)
    if not candidates:
        return None

    if strategy == "earliest":
        return min(candidates).date()
    elif strategy == "latest":
        return max(candidates).date()
    else:  # "first" (default)
        return candidates[0].date()


# ---------------------------------------------------------------------------
# Text extraction from PDF
# ---------------------------------------------------------------------------

def extract_embedded_text(pdf_path: str) -> str:
    """Extract text embedded in the PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts)
    except Exception as e:
        print(f"  [warn] Could not read embedded text from {pdf_path}: {e}", file=sys.stderr)
        return ""


def extract_ocr_text(pdf_path: str, max_pages: int = 5, dpi: int = 200) -> str:
    """Render PDF pages as images and run Tesseract OCR on them."""
    pytesseract, Image, io = _import_ocr()
    parts = []
    try:
        doc = fitz.open(pdf_path)
        # Only scan first max_pages pages to keep things fast
        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            parts.append(text)
        doc.close()
    except Exception as e:
        print(f"  [warn] OCR failed for {pdf_path}: {e}", file=sys.stderr)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core rename logic
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    use_ocr: bool = True,
    use_embedded: bool = True,
    strategy: str = "first",
) -> tuple[bool, str]:
    """
    Attempt to find a date in the PDF and rename it.

    Returns (success, message).
    """
    name = pdf_path.name

    # Skip files already prefixed with a date
    if _ALREADY_DATED.match(name):
        return False, f"Skipped (already dated): {pdf_path}"

    text = ""

    # 1. Try embedded text first (fast)
    if use_embedded:
        text = extract_embedded_text(str(pdf_path))
        if verbose and text.strip():
            snippet = " ".join(text.split())[:200]
            print(f"  [embedded text] {snippet}…")

    # 2. Fall back to OCR if embedded text has no date-like content
    found_date = extract_dates_from_text(text, strategy=strategy)

    if found_date is None and use_ocr:
        if verbose:
            print("  [info] No date in embedded text; running OCR…")
        ocr_text = extract_ocr_text(str(pdf_path))
        if verbose and ocr_text.strip():
            snippet = " ".join(ocr_text.split())[:200]
            print(f"  [OCR text] {snippet}…")
        found_date = extract_dates_from_text(ocr_text, strategy=strategy)

    if found_date is None:
        return False, f"No date found: {pdf_path}"

    # Build new filename
    date_prefix = found_date.strftime("%Y-%m-%d")
    new_name = f"{date_prefix}_{name}"
    new_path = pdf_path.parent / new_name

    # Handle collisions
    if new_path.exists() and new_path != pdf_path:
        stem = pdf_path.stem
        suffix = pdf_path.suffix
        counter = 1
        while new_path.exists():
            new_name = f"{date_prefix}_{stem}_{counter}{suffix}"
            new_path = pdf_path.parent / new_name
            counter += 1

    action = "Would rename" if dry_run else "Renamed"
    msg = f"{action}: {pdf_path.name}  →  {new_name}"

    if not dry_run:
        try:
            pdf_path.rename(new_path)
        except OSError as e:
            return False, f"Error renaming {pdf_path}: {e}"

    return True, msg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_pdfs(targets: list[str], recursive: bool) -> list[Path]:
    """Expand files and directories into a list of PDF paths."""
    pdfs = []
    for target in targets:
        p = Path(target)
        if p.is_file():
            if p.suffix.lower() == ".pdf":
                pdfs.append(p)
            else:
                print(f"[warn] Not a PDF, skipping: {p}", file=sys.stderr)
        elif p.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            pdfs.extend(sorted(p.glob(pattern)))
        else:
            print(f"[warn] Path not found: {p}", file=sys.stderr)
    return pdfs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="PDF_OR_DIR",
        help="PDF files or directories to process (default: current directory)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively scan directories",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show renames without executing them",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show extracted text snippets and OCR progress",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Do not fall back to OCR; only use embedded PDF text",
    )
    parser.add_argument(
        "--no-embedded",
        action="store_true",
        help="Skip embedded text extraction; always use OCR",
    )

    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--earliest",
        action="store_const",
        dest="strategy",
        const="earliest",
        help="Use the chronologically earliest date found",
    )
    strategy_group.add_argument(
        "--latest",
        action="store_const",
        dest="strategy",
        const="latest",
        help="Use the chronologically latest date found",
    )
    parser.set_defaults(strategy="first")

    args = parser.parse_args()

    targets = args.targets if args.targets else ["."]
    pdfs = collect_pdfs(targets, recursive=args.recursive)

    if not pdfs:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[dry-run mode — no files will be changed]\n")

    renamed = 0
    skipped = 0
    errors = 0

    for pdf in pdfs:
        if args.verbose:
            print(f"\nProcessing: {pdf}")
        success, msg = process_pdf(
            pdf,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_ocr=not args.no_ocr,
            use_embedded=not args.no_embedded,
            strategy=args.strategy,
        )
        print(msg)
        if success:
            renamed += 1
        elif msg.startswith("Error"):
            errors += 1
        else:
            skipped += 1

    total = renamed + skipped + errors
    print(f"\nDone. {renamed}/{total} file(s) {'would be ' if args.dry_run else ''}renamed, "
          f"{skipped} skipped, {errors} error(s).")


if __name__ == "__main__":
    main()
