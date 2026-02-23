from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
from readability import Document
import fitz  # pymupdf
from pptx import Presentation
import io


def extract_text_from_txt(path: Path) -> Dict[str, Any]:
    """Extract plain text from a .txt file.

    Returns a dict with keys: `text`, `pages` (1), `source` (path str).
    """
    text = None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        # fallback to latin1
        text = path.read_text(encoding="latin-1")
    return {"text": text, "pages": 1, "source": str(path)}


def extract_text_from_url(url: str) -> Dict[str, Any]:
    """Fetch a URL and extract the main article text using readability + BeautifulSoup.

    Returns dict with keys: `text`, `title`, `html`, `source`.
    """
    resp = requests.get(url, timeout=15, headers={"User-Agent": "ingestion-bot/1.0"})
    resp.raise_for_status()
    html = resp.text
    doc = Document(html)
    summary_html = doc.summary()
    title = doc.title()
    soup = BeautifulSoup(summary_html, "html.parser")
    text = soup.get_text(separator="\n")
    return {"text": text, "title": title, "html": html, "source": url}


def extract_text_from_pdf(path: Path, use_ocr: bool = False) -> Dict[str, Any]:
    """Extract text from a PDF file using PyMuPDF (fitz).

    Args:
        path: Path to PDF file.
        use_ocr: If True, attempt OCR on images (requires pytesseract + tesseract).
                 Defaults to False (text-only extraction).

    Returns dict with keys: `text`, `pages`, `source`.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF {path}: {e}")

    text_parts = []
    page_count = len(doc)  # Capture page count before closing
    for page_num, page in enumerate(doc):
        try:
            text = page.get_text()
            if not text.strip() and use_ocr:
                # Attempt OCR on images in this page if text extraction is empty
                try:
                    import pytesseract
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes(output="png")
                    import PIL.Image
                    img = PIL.Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(img)
                except Exception as ocr_err:
                    text = f"[OCR failed on page {page_num + 1}: {ocr_err}]"
            text_parts.append(text)
        except Exception as e:
            text_parts.append(f"[Error extracting page {page_num + 1}: {e}]")

    full_text = "\n".join(text_parts)
    doc.close()
    return {"text": full_text, "pages": page_count, "source": str(path)}


def extract_text_from_pptx(path: Path) -> Dict[str, Any]:
    """Extract text from a PowerPoint file (.pptx).

    Returns dict with keys: `text`, `slides`, `source`.
    """
    try:
        prs = Presentation(path)
    except Exception as e:
        raise ValueError(f"Failed to open PPTX {path}: {e}")

    slide_texts = []
    for slide_num, slide in enumerate(prs.slides):
        slide_text_parts = []
        for shape in slide.shapes:
            try:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text_parts.append(shape.text)
            except Exception:
                pass
        slide_texts.append("\n".join(slide_text_parts) if slide_text_parts else f"[Slide {slide_num + 1} - no text extracted]")

    full_text = "\n\n".join(slide_texts)
    return {"text": full_text, "slides": len(prs.slides), "source": str(path)}
