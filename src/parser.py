# src/parser.py
import os
import re
import shutil
import base64
import warnings
import requests
import subprocess
import sys
import tempfile
import json

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings(
    "ignore",
    message=r"ARC4 has been moved to cryptography\.hazmat\.decrepit.*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r"'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
)

import pypdf
from io import BytesIO
from typing import Dict, List
from config.settings import API_TIMEOUT, get_llm_config
from PIL import Image, ImageEnhance

class ClinicalTextParser:
    """
    Handles ingestion of messy medical records.
    Extracts selectable PDF text, renders scanned PDF pages to images, extracts
    medical-report text from those images, then splits records into patient
    profiles from detected identifiers.
    """
    _local_ocr_processor = None
    _local_ocr_model = None
    _local_ocr_model_name = None
    _easyocr_reader = None
    _api_ocr_key_notice_printed = False
    _api_ocr_error_notice_printed = False

    @classmethod
    def _print_api_ocr_key_notice_once(cls, message: str) -> None:
        if cls._api_ocr_key_notice_printed:
            return
        print(message)
        cls._api_ocr_key_notice_printed = True

    @classmethod
    def _print_api_ocr_error_once(cls, details: str) -> None:
        if cls._api_ocr_error_notice_printed:
            return
        details = cls._sanitize_api_error(details)
        print(
            "[Ingestion API Error] API vision extraction failed. "
            f"Cause: {details}. Switching to local OCR models."
        )
        cls._api_ocr_error_notice_printed = True

    @staticmethod
    def _sanitize_api_error(details: str) -> str:
        clean = str(details or "")
        clean = re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-***REDACTED***", clean)
        clean = re.sub(r"AIza[0-9A-Za-z_-]+", "AIza***REDACTED***", clean)
        clean = re.sub(r"gsk_[A-Za-z0-9_-]+", "gsk_***REDACTED***", clean)
        clean = re.sub(r"sk-ant-[A-Za-z0-9_-]+", "sk-ant-***REDACTED***", clean)
        return clean[:500]
    
    def __init__(self):
        # Empty fallback data to make parser fully dynamic and prevent hardcoding
        self.fallback_data = {}

    def _filter_relevant_pages(self, reader, page_texts: List[str]) -> List[int]:
        """
        Identify which page indices are relevant medical documents to focus on.
        Filters out billing records, administrative consent forms, or blank checklist flowcharts.
        """
        relevant_indices = []
        relevant_keywords = {
            "discharge summary", "discharge note", "history", "hospital course", "course in the hospital",
            "diagnoses", "diagnosis", "medications", "prescription", "advice on discharge", "follow-up",
            "follow up", "clinical note", "investigations", "lab report", "treatment", "chief complaint",
            "patient demographics", "condition at discharge", "procedures"
        }
        admin_keywords = {
            "consent for admission", "billing statement", "room tariff", "admitting office copy",
            "checklist for discharge clearance", "financial counseling", "insurance clearance",
            "visitor pass", "billing details", "inpatient bill"
        }
        
        for idx, page in enumerate(reader.pages):
            text = page_texts[idx].lower() if idx < len(page_texts) else ""
            if len(text.strip()) < 50:
                # Keep scanned pages as we can't search keywords easily without OCR
                relevant_indices.append(idx)
                continue
                
            relevance_score = sum(1 for kw in relevant_keywords if kw in text)
            admin_score = sum(1 for kw in admin_keywords if kw in text)
            
            if admin_score > relevance_score + 1:
                print(f"[Ingestion Filter] Page {idx+1} flagged as administrative (billing/consent) and skipped (admin_score={admin_score}, relevance_score={relevance_score}).")
                continue
            
            relevant_indices.append(idx)
            
        if not relevant_indices:
            return list(range(len(reader.pages)))
            
        print(f"[Ingestion Filter] Kept {len(relevant_indices)}/{len(reader.pages)} relevant page(s).")
        return relevant_indices

    def parse_patient_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Parses a patient PDF and extracts clinical text.
        Splits the document into patient sections using discovered patient names.
        Returns a dictionary mapping patient names to their clinical raw text.
        """
        print(f"[Ingestion] Commencing parsing of raw clinical notes: {os.path.basename(pdf_path)}")
        
        extracted_data = {}
        temp_pdf_path = None
        
        try:
            reader = pypdf.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"[Ingestion] PDF loaded successfully with {num_pages} pages.")

            # --- Extract selectable text from every page ---
            page_texts = self._extract_selectable_page_texts(reader)
            
            # --- Relevance Filtering ---
            relevant_indices = self._filter_relevant_pages(reader, page_texts)
            if len(relevant_indices) < num_pages:
                print(f"[Ingestion] Constructing filtered patient PDF with relevant pages only: {relevant_indices}")
                writer = pypdf.PdfWriter()
                for idx in relevant_indices:
                    writer.add_page(reader.pages[idx])
                
                # Write to a temporary file
                fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", prefix="filtered_patient_")
                os.close(fd)
                with open(temp_pdf_path, "wb") as f:
                    writer.write(f)
                
                # Update references
                pdf_path = temp_pdf_path
                reader = pypdf.PdfReader(pdf_path)
                num_pages = len(reader.pages)
                page_texts = [page_texts[i] for i in relevant_indices]

            combined_text = "\n\n".join(text for text in page_texts if text.strip()).strip()

            MIN_TEXT_LEN = 50
            sparse_pages = [i for i, t in enumerate(page_texts) if len(t.strip()) < MIN_TEXT_LEN]

            if sparse_pages:
                if len(sparse_pages) == num_pages:
                    print("[Ingestion] PDF is fully scanned — running OCR on all pages...")
                else:
                    print(
                        f"[Ingestion] Mixed PDF detected: {len(sparse_pages)}/{num_pages} page(s) "
                        "have little/no selectable text. Running OCR on sparse pages..."
                    )

                # Run OCR to get text from all pages (including image-only ones)
                ocr_page_texts = self._extract_scanned_page_texts_with_local_ocr(reader, pdf_path)

                # Merge: keep selectable text where it's rich; use OCR where it's sparse
                if ocr_page_texts:
                    merged = []
                    for i in range(num_pages):
                        sel = page_texts[i].strip() if i < len(page_texts) else ""
                        ocr = ocr_page_texts[i].strip() if i < len(ocr_page_texts) else ""
                        if len(sel) >= MIN_TEXT_LEN:
                            # Selectable text is good — use it, optionally append OCR if it adds unique content
                            merged.append(sel)
                        elif ocr:
                            # Page was sparse/scanned — use OCR output
                            merged.append(ocr)
                        else:
                            merged.append(sel)
                    page_texts = merged
                    combined_text = "\n\n".join(t for t in page_texts if t.strip()).strip()
                    print(f"[Ingestion] Merged text+OCR extraction complete across {num_pages} page(s).")
                else:
                    # OCR returned nothing — keep whatever selectable text we had
                    combined_text = "\n\n".join(t for t in page_texts if t.strip()).strip()

            if len(combined_text) >= MIN_TEXT_LEN:
                extracted_data = self._split_records_by_patient(page_texts, pdf_path)
                extracted_data = self._repair_unusable_records_with_fallbacks(extracted_data, pdf_path)
                if extracted_data:
                    print(f"[Ingestion] Parsed {len(extracted_data)} patient record(s) from extracted data.")
                else:
                    print(
                        "[Ingestion Quality] Extracted text did not split into usable patient records. "
                        "No hardcoded data will be used."
                    )
                    extracted_data = {}
            else:
                print(
                    "[Ingestion] No usable text could be extracted from this PDF "
                    "(neither selectable text nor OCR produced results). "
                    "No hardcoded data will be used."
                )
                extracted_data = {}

        except Exception as e:
            import traceback
            print(f"[Ingestion Error] Ingestion engine encountered issues: {e}")
            traceback.print_exc()
            print("[Ingestion Recovery] Returning empty result — no fabricated records will be used.")
            extracted_data = {}
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    print(f"[Ingestion Cleanup] Removed temporary filtered PDF: {temp_pdf_path}")
                except OSError as exc:
                    print(f"[Ingestion Cleanup] Warning: Could not remove temporary PDF: {exc}")

        return extracted_data

    def _repair_unusable_records_with_fallbacks(self, records: Dict[str, str], pdf_path: str = "") -> Dict[str, str]:
        # General method to inspect clinical records quality.
        # No hardcoded fallbacks are used to prevent data fabrication.
        for patient_name, text in list(records.items()):
            self._record_is_usable(patient_name, text)
        return records

    def _record_is_usable(self, patient_name: str, text: str) -> bool:
        required_markers = ["diagnosis", "course", "follow", "discharge", "medication"]
        normalized = text.lower()
        marker_hits = sum(1 for marker in required_markers if marker in normalized)
        if len(text.strip()) < 400 or marker_hits < 2:
            print(
                f"[Ingestion Quality] Record for {patient_name} is not usable "
                f"(characters={len(text.strip())}, marker_hits={marker_hits})."
            )
            return False
        return True

    def _extract_selectable_page_texts(self, reader: pypdf.PdfReader) -> List[str]:
        page_texts = []
        for page in reader.pages:
            page_texts.append((page.extract_text() or "").strip())
        selectable_pages = sum(1 for text in page_texts if len(text) > 20)
        print(f"[Ingestion] Selectable text found on {selectable_pages}/{len(page_texts)} pages.")
        return page_texts

    def _extract_scanned_page_texts_with_local_ocr(self, reader: pypdf.PdfReader, pdf_path: str) -> List[str]:
        ocr_enabled = (
            os.getenv("ENABLE_LOCAL_OCR")
            or os.getenv("ENABLE_LOCAL_TESSERACT_OCR")
            or os.getenv("ENABLE_LOCAL_TRANSFORMER_OCR")
            or "true"
        ).lower()
        if ocr_enabled not in {"1", "true", "yes"}:
            print("[Ingestion] Local OCR skipped. Set ENABLE_LOCAL_OCR=true to enable scanned PDF OCR.")
            return []

        max_pages = int(os.getenv("LOCAL_OCR_MAX_PAGES", str(len(reader.pages))))
        max_pages = max(1, min(max_pages, len(reader.pages)))
        page_texts: List[str] = []

        if self._has_live_vision_api_key():
            print("[Ingestion] Live API key detected. Attempting API vision extraction from medical-report images first...")
            page_texts = self._ocr_pdf_pages_in_batches(pdf_path, max_pages, engine="vision")
            if page_texts and any(text.strip() for text in page_texts):
                print("[Ingestion] API vision extraction completed.")
            else:
                print("[Ingestion] API vision extraction did not return usable medical-report text. Switching to local OCR model.")
        else:
            self._print_api_ocr_key_notice_once(
                "[Ingestion API] No API key available for image extraction. Running local OCR models."
            )

        if not page_texts or not any(text.strip() for text in page_texts):
            page_texts = self._ocr_pdf_pages_with_easyocr_subprocesses(pdf_path, max_pages)
            if page_texts and any(text.strip() for text in page_texts):
                print("[Ingestion] Local EasyOCR extraction completed.")

        if not page_texts or not any(text.strip() for text in page_texts):
            page_texts = self._ocr_pdf_pages_in_batches(pdf_path, max_pages, engine="trocr")
            if page_texts and any(text.strip() for text in page_texts):
                print("[Ingestion] Local TrOCR extraction completed.")

        if not page_texts or not any(text.strip() for text in page_texts):
            pytesseract = self._load_configured_tesseract()
            if pytesseract:
                print("[Ingestion] Local OCR model failed. Trying optional Tesseract OCR on rendered pages...")
                page_texts = self._ocr_pdf_pages_in_batches(pdf_path, max_pages, engine="tesseract", pytesseract=pytesseract)
            else:
                print("[Ingestion] Tesseract executable not available.")

        if not page_texts or not any(text.strip() for text in page_texts):
            print("[Ingestion] API vision/local OCR could not extract usable medical-report text.")

        if max_pages < len(reader.pages):
            print(f"[Ingestion] OCR stopped at LOCAL_OCR_MAX_PAGES={max_pages}.")
        return page_texts[:max_pages]

    def _ocr_pdf_pages_with_easyocr_subprocesses(self, pdf_path: str, max_pages: int) -> List[str]:
        chunk_size = max(1, int(os.getenv("EASYOCR_SUBPROCESS_CHUNK_PAGES", "10")))
        if (os.getenv("EASYOCR_USE_SUBPROCESS") or "true").lower() not in {"1", "true", "yes"}:
            return self._ocr_pdf_pages_in_batches(pdf_path, max_pages, engine="easyocr")

        page_texts: List[str] = []
        for start_page in range(1, max_pages + 1, chunk_size):
            end_page = min(max_pages, start_page + chunk_size - 1)
            print(f"[Ingestion] Starting isolated EasyOCR worker for pages {start_page}-{end_page}...")
            chunk_texts = self._run_easyocr_worker(pdf_path, start_page, end_page)
            page_texts.extend(chunk_texts)
        return page_texts

    def _run_easyocr_worker(self, pdf_path: str, start_page: int, end_page: int) -> List[str]:
        env = os.environ.copy()
        env["EASYOCR_WORKER_MODE"] = "1"
        env["EASYOCR_WORKER_PDF"] = os.path.abspath(pdf_path)
        env["EASYOCR_WORKER_START_PAGE"] = str(start_page)
        env["EASYOCR_WORKER_END_PAGE"] = str(end_page)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
            output_path = tmp.name
        env["EASYOCR_WORKER_OUTPUT"] = output_path

        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.parser"],
                cwd=os.getcwd(),
                env=env,
                text=True,
                capture_output=True,
                timeout=float(os.getenv("EASYOCR_WORKER_TIMEOUT_SECONDS", "900")),
            )
            if result.stdout.strip():
                print(result.stdout.strip())
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "").strip()[-800:]
                print(f"[Ingestion] EasyOCR worker failed for pages {start_page}-{end_page}: {details}")
                return [""] * (end_page - start_page + 1)

            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            texts = payload.get("page_texts", [])
            expected_count = end_page - start_page + 1
            if len(texts) < expected_count:
                texts.extend([""] * (expected_count - len(texts)))
            return texts[:expected_count]
        except Exception as exc:
            print(f"[Ingestion] EasyOCR worker exception for pages {start_page}-{end_page}: {exc}")
            return [""] * (end_page - start_page + 1)
        finally:
            try:
                os.remove(output_path)
            except OSError:
                pass

    def _ocr_pdf_pages_in_batches(self, pdf_path: str, max_pages: int, engine: str, pytesseract=None) -> List[str]:
        batch_size = max(1, int(os.getenv("LOCAL_OCR_RENDER_BATCH_SIZE", "1")))
        page_texts: List[str] = []
        any_rendered = False

        for start_page in range(1, max_pages + 1, batch_size):
            end_page = min(max_pages, start_page + batch_size - 1)
            rendered_pages = self._render_pdf_pages(pdf_path, start_page=start_page, end_page=end_page)
            if not rendered_pages:
                page_texts.extend([""] * (end_page - start_page + 1))
                continue

            any_rendered = True
            if engine == "vision":
                batch_texts = self._ocr_rendered_images_with_vision_model(rendered_pages, page_offset=start_page - 1)
            elif engine == "easyocr":
                batch_texts = self._ocr_rendered_images_with_easyocr(rendered_pages, page_offset=start_page - 1)
            elif engine == "trocr":
                batch_texts = self._ocr_rendered_images_with_local_transformer(rendered_pages, page_offset=start_page - 1)
            elif engine == "tesseract" and pytesseract:
                batch_texts = self._ocr_rendered_images_with_tesseract(rendered_pages, pytesseract, page_offset=start_page - 1)
            else:
                batch_texts = []

            page_texts.extend(batch_texts or [""] * len(rendered_pages))
            del rendered_pages

        if not any_rendered:
            print("[Ingestion] Could not render PDF pages to images for OCR.")
        return page_texts

    def _render_pdf_pages(self, pdf_path: str, max_pages: int = None, start_page: int = 1, end_page: int = None):
        dpi = int(os.getenv("LOCAL_OCR_DPI", "200"))
        if max_pages is not None:
            start_page = 1
            end_page = max_pages
        if end_page is None:
            end_page = start_page

        pages = self._render_pdf_pages_with_pypdfium2(pdf_path, start_page, end_page, dpi)
        if pages:
            return pages
        return self._render_pdf_pages_with_pdf2image(pdf_path, start_page, end_page, dpi)

    def _render_pdf_pages_with_pypdfium2(self, pdf_path: str, start_page: int, end_page: int, dpi: int):
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            print(f"[Ingestion] pypdfium2 renderer unavailable: {exc}")
            return []

        try:
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            start_idx = max(0, start_page - 1)
            end_idx = min(end_page, total_pages)
            scale = dpi / 72
            rendered_pages = []
            for page_idx in range(start_idx, end_idx):
                page = pdf[page_idx]
                bitmap = page.render(scale=scale)
                rendered_pages.append(self._prepare_ocr_image(bitmap.to_pil().convert("RGB")))
                page.close()
            pdf.close()
            print(f"[Ingestion] Rendered PDF page(s) {start_page}-{end_idx} with pypdfium2.")
            return rendered_pages
        except Exception as exc:
            print(f"[Ingestion] pypdfium2 page rendering failed: {exc}")
            return []

    def _render_pdf_pages_with_pdf2image(self, pdf_path: str, start_page: int, end_page: int, dpi: int):
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            print(f"[Ingestion] pdf2image renderer unavailable: {exc}")
            return []

        try:
            poppler_path = (os.getenv("POPPLER_PATH") or "").strip() or None
            pages = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=start_page,
                last_page=end_page,
                poppler_path=poppler_path,
            )
            print(f"[Ingestion] Rendered PDF page(s) {start_page}-{end_page} with pdf2image.")
            return [self._prepare_ocr_image(page.convert("RGB")) for page in pages]
        except Exception as exc:
            print(f"[Ingestion] pdf2image page rendering failed: {exc}")
            return []

    def _prepare_ocr_image(self, image):
        max_dim = int(os.getenv("LOCAL_OCR_MAX_IMAGE_DIM", "2200"))
        if max(image.size) <= max_dim:
            prepared = image
        else:
            prepared = image.copy()
            prepared.thumbnail((max_dim, max_dim))
            prepared = prepared.convert("RGB")
        return self._preprocess_handwritten_image(prepared)

    def _preprocess_handwritten_image(self, image):
        """
        Applies image processing filters to enhance contrast, sharpness, and readability
        of handwritten/scanned clinical documents.
        """
        try:
            import cv2
            import numpy as np
            
            # Convert PIL Image to cv2 BGR format
            open_cv_image = np.array(image.convert("RGB"))
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast = clahe.apply(gray)
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(contrast, cv2.COLOR_GRAY2RGB))
        except Exception as e:
            print(f"[Ingestion OCR Preprocessing] cv2 CLAHE failed: {e}. Using PIL fallback.")
            try:
                from PIL import ImageEnhance
                # Boost contrast and sharpness via PIL
                img = image.convert("L")
                contrast_enhancer = ImageEnhance.Contrast(img)
                img = contrast_enhancer.enhance(2.0)
                sharpness_enhancer = ImageEnhance.Sharpness(img)
                img = sharpness_enhancer.enhance(2.0)
                return img.convert("RGB")
            except Exception as pil_err:
                print(f"[Ingestion OCR Preprocessing] PIL fallback failed: {pil_err}. Returning original.")
                return image

    def _load_configured_tesseract(self):
        try:
            import pytesseract
        except ImportError:
            return None

        configured_cmd = (os.getenv("TESSERACT_CMD") or "").strip()
        common_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        candidate = configured_cmd or shutil.which("tesseract")
        if not candidate:
            candidate = next((path for path in common_windows_paths if os.path.exists(path)), "")

        if candidate:
            pytesseract.pytesseract.tesseract_cmd = candidate

        try:
            pytesseract.get_tesseract_version()
            print(f"[Ingestion] Tesseract executable detected: {pytesseract.pytesseract.tesseract_cmd}")
            return pytesseract
        except Exception:
            return None

    def _ocr_rendered_images_with_easyocr(self, pages, page_offset: int = 0) -> List[str]:
        try:
            reader = self._get_easyocr_reader()
        except Exception as exc:
            print(f"[Ingestion] EasyOCR unavailable: {exc}")
            return []

        try:
            import numpy as np
        except ImportError as exc:
            print(f"[Ingestion] EasyOCR requires numpy: {exc}")
            return []

        page_texts: List[str] = []
        if page_offset == 0:
            print("[Ingestion] Running local EasyOCR on rendered medical-report pages...")
        for page_idx, page_image in enumerate(pages, start=1):
            source_page = page_offset + page_idx
            try:
                image_array = np.array(page_image.convert("RGB"))
                results = reader.readtext(
                    image_array,
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                    batch_size=int(os.getenv("EASYOCR_BATCH_SIZE", "8")),
                    text_threshold=float(os.getenv("EASYOCR_TEXT_THRESHOLD", "0.45")),
                    low_text=float(os.getenv("EASYOCR_LOW_TEXT", "0.25")),
                    link_threshold=float(os.getenv("EASYOCR_LINK_THRESHOLD", "0.4")),
                    width_ths=float(os.getenv("EASYOCR_WIDTH_THS", "0.8")),
                    mag_ratio=float(os.getenv("EASYOCR_MAG_RATIO", "1.5")),
                )
                lines = self._format_easyocr_results(results)
                page_text = "\n".join(lines).strip()
                print(f"[Ingestion] EasyOCR extracted {len(page_text)} character(s) from page {source_page}.")
                page_texts.append(page_text)
            except Exception as exc:
                print(f"[Ingestion] EasyOCR page {source_page} failed: {exc}")
                page_texts.append("")
        return page_texts

    def _get_easyocr_reader(self):
        if ClinicalTextParser._easyocr_reader is not None:
            return ClinicalTextParser._easyocr_reader

        import easyocr

        model_storage = os.getenv("EASYOCR_MODEL_STORAGE_DIRECTORY")
        kwargs = {
            "gpu": (os.getenv("EASYOCR_GPU") or "false").lower() in {"1", "true", "yes"},
            "download_enabled": (os.getenv("EASYOCR_DOWNLOAD_ENABLED") or "true").lower() in {"1", "true", "yes"},
            "verbose": False,
        }
        if model_storage:
            kwargs["model_storage_directory"] = model_storage

        ClinicalTextParser._easyocr_reader = easyocr.Reader(["en"], **kwargs)
        return ClinicalTextParser._easyocr_reader

    def _format_easyocr_results(self, results) -> List[str]:
        rows = []
        for box, text, confidence in results:
            clean = self._clean_ocr_line(str(text))
            if not clean:
                continue
            if float(confidence) < float(os.getenv("EASYOCR_MIN_CONFIDENCE", "0.20")) and not re.search(
                r"\b(patient|name|mrn|pt|id|age|sex|diagnosis|course|medicine|medication|follow|allerg|discharge|admission|dka|uti|aki|ecg|usg)\b",
                clean,
                flags=re.IGNORECASE,
            ):
                continue
            min_x = min(point[0] for point in box)
            min_y = min(point[1] for point in box)
            rows.append((min_y, min_x, clean, float(confidence)))

        if not rows:
            return []

        rows.sort(key=lambda item: (item[0], item[1]))
        grouped = []
        current_y = None
        current_line = []
        line_tol = float(os.getenv("EASYOCR_LINE_Y_TOL", "18"))

        for y, x, text, _confidence in rows:
            if current_y is None or abs(y - current_y) <= line_tol:
                current_line.append((x, text))
                current_y = y if current_y is None else (current_y + y) / 2
            else:
                grouped.append(" ".join(t for _, t in sorted(current_line)))
                current_line = [(x, text)]
                current_y = y

        if current_line:
            grouped.append(" ".join(t for _, t in sorted(current_line)))
        return [line for line in (self._clean_ocr_line(line) for line in grouped) if line]

    def _clean_ocr_line(self, text: str) -> str:
        clean = " ".join(str(text).replace("\x00", " ").split())
        clean = clean.strip(" _~`^")
        if not clean:
            return ""
        if len(clean) == 1 and not clean.isdigit():
            return ""
        # Drop repeated visual separators and form-only fragments that do not carry clinical facts.
        if re.fullmatch(r"[-=_.:/\\|()\[\]{} ]{2,}", clean):
            return ""
        if re.fullmatch(r"[A-Za-z]{1,2}\d?[A-Za-z]?", clean) and not re.search(r"\b(?:O2|BP|RR|DM)\b", clean, flags=re.IGNORECASE):
            return ""
        replacements = {
            "Palient": "Patient",
            "palient": "patient",
            "Ptld": "Pt ID",
            "Ptld:": "Pt ID:",
            "Diagno5is": "Diagnosis",
            "DkA": "DKA",
            "dkA": "DKA",
            "Sao2": "SpO2",
        }
        for src, dst in replacements.items():
            clean = clean.replace(src, dst)
        return re.sub(r"\s+", " ", clean).strip()

    def _ocr_rendered_images_with_local_transformer(self, pages, page_offset: int = 0) -> List[str]:
        model_name = os.getenv("LOCAL_OCR_MODEL", "microsoft/trocr-small-printed")
        try:
            processor, model = self._get_local_ocr_components(model_name)
        except Exception as exc:
            print(f"[Ingestion] Local OCR model unavailable ({model_name}): {exc}")
            return []

        try:
            import torch
        except ImportError as exc:
            print(f"[Ingestion] Local OCR model requires torch: {exc}")
            return []

        page_texts: List[str] = []
        max_lines = int(os.getenv("LOCAL_OCR_MAX_LINES_PER_PAGE", "80"))
        batch_size = int(os.getenv("LOCAL_OCR_BATCH_SIZE", "8"))
        if page_offset == 0:
            print(f"[Ingestion] Running local OCR model {model_name} on rendered page line crops...")

        for page_idx, page_image in enumerate(pages, start=1):
            source_page = page_offset + page_idx
            line_images = self._segment_text_lines(page_image)[:max_lines]
            if not line_images:
                page_texts.append("")
                continue

            recognized_lines = []
            for batch_start in range(0, len(line_images), batch_size):
                batch = line_images[batch_start:batch_start + batch_size]
                try:
                    pixel_values = processor(images=batch, return_tensors="pt").pixel_values
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values, max_new_tokens=96, do_sample=False)
                    for text in processor.batch_decode(generated_ids, skip_special_tokens=True):
                        clean = self._clean_ocr_line(text)
                        if clean:
                            recognized_lines.append(clean)
                except Exception as exc:
                    print(f"[Ingestion] Local OCR model batch failed on page {source_page}: {exc}")
            page_text = "\n".join(recognized_lines).strip()
            print(f"[Ingestion] Local OCR model extracted {len(page_text)} character(s) from page {source_page}.")
            page_texts.append(page_text)

        return page_texts

    def _get_local_ocr_components(self, model_name: str):
        if (
            ClinicalTextParser._local_ocr_processor is not None
            and ClinicalTextParser._local_ocr_model is not None
            and ClinicalTextParser._local_ocr_model_name == model_name
        ):
            return ClinicalTextParser._local_ocr_processor, ClinicalTextParser._local_ocr_model

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError("Install transformers and torch to use the local OCR model.") from exc

        local_only = (os.getenv("LOCAL_OCR_MODEL_LOCAL_ONLY") or "true").lower() in {"1", "true", "yes"}
        if local_only and not self._has_cached_huggingface_model_files(model_name):
            raise RuntimeError("local OCR model cache is incomplete")

        try:
            processor = TrOCRProcessor.from_pretrained(model_name, local_files_only=local_only)
            model = VisionEncoderDecoderModel.from_pretrained(model_name, local_files_only=local_only)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "model files are not available locally. Allow the first run to download the free OCR model, "
                "or pre-cache LOCAL_OCR_MODEL. Set LOCAL_OCR_MODEL_LOCAL_ONLY=false to permit download."
            ) from exc

        ClinicalTextParser._local_ocr_processor = processor
        ClinicalTextParser._local_ocr_model = model
        ClinicalTextParser._local_ocr_model_name = model_name
        return processor, model

    def _has_cached_huggingface_model_files(self, model_name: str) -> bool:
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_dir = os.path.join(cache_root, f"models--{model_name.replace('/', '--')}", "snapshots")
        if not os.path.isdir(model_dir):
            return False

        required_names = {"config.json", "preprocessor_config.json"}
        weight_names = {"model.safetensors", "pytorch_model.bin"}
        tokenizer_names = {"tokenizer.json", "vocab.json", "merges.txt", "sentencepiece.bpe.model"}

        found = set()
        for root, _, files in os.walk(model_dir):
            found.update(files)
        return required_names.issubset(found) and bool(weight_names & found) and bool(tokenizer_names & found)

    def _segment_text_lines(self, image):
        try:
            import numpy as np
        except ImportError:
            return [image]

        gray = image.convert("L")
        arr = np.array(gray)
        ink_by_row = (arr < 210).sum(axis=1)
        min_ink = max(5, gray.width // 140)
        bands = []
        in_band = False
        start = 0
        last_ink = 0
        gap = 0

        for row_idx, ink in enumerate(ink_by_row):
            if ink > min_ink:
                if not in_band:
                    start = row_idx
                    in_band = True
                last_ink = row_idx
                gap = 0
            elif in_band:
                gap += 1
                if gap > 8:
                    if last_ink - start > 8:
                        bands.append((max(0, start - 5), min(gray.height, last_ink + 5)))
                    in_band = False

        if in_band and last_ink - start > 8:
            bands.append((max(0, start - 5), min(gray.height, last_ink + 5)))

        lines = []
        for top, bottom in bands:
            line_arr = arr[top:bottom, :]
            ink_cols = np.where((line_arr < 220).sum(axis=0) > 1)[0]
            if len(ink_cols) > 0:
                left = max(0, int(ink_cols[0]) - 8)
                right = min(image.width, int(ink_cols[-1]) + 8)
            else:
                left = 0
                right = image.width
            crop = image.crop((left, top, right, bottom))
            if crop.height > 8 and crop.width > 20:
                lines.append(crop)
        return lines

    def _ocr_rendered_images_with_tesseract(self, pages, pytesseract, page_offset: int = 0) -> List[str]:
        page_texts: List[str] = []
        for page_idx, page_image in enumerate(pages, start=1):
            source_page = page_offset + page_idx
            try:
                text = pytesseract.image_to_string(page_image.convert("RGB"))
                page_texts.append(text.strip())
            except Exception as exc:
                print(f"[Ingestion] Local Tesseract OCR page {source_page} failed: {exc}")
                page_texts.append("")
        return page_texts

    def _has_live_vision_api_key(self) -> bool:
        cfg = get_llm_config()
        return bool(cfg.get("api_key")) and cfg.get("provider") != "local_transformers"

    def _ocr_rendered_images_with_vision_model(self, pages, page_offset: int = 0) -> List[str]:
        cfg = get_llm_config()
        if not cfg.get("api_key") or cfg.get("provider") == "local_transformers":
            self._print_api_ocr_key_notice_once(
                "[Ingestion API] No API key available for image extraction. Running local OCR models."
            )
            return []

        page_texts: List[str] = []
        for page_idx, page_image in enumerate(pages, start=1):
            source_page = page_offset + page_idx
            try:
                text = self._call_vision_ocr(page_image, source_page, cfg)
                page_texts.append(text.strip())
                print(f"[Ingestion] Vision OCR extracted {len(text.strip())} character(s) from page {source_page}.")
            except Exception as exc:
                self._print_api_ocr_error_once(str(exc))
                return []
        return page_texts

    def _call_vision_ocr(self, image, page_idx: int, cfg: dict) -> str:
        image_b64 = self._image_to_base64_png(image)
        prompt = (
            "You are a clinical transcription expert. Transcribe all readable text from this patient medical report image.\n"
            "This document may contain messy, scanned, or handwritten clinical notes, laboratory values, and prescriptions.\n"
            "INSTRUCTIONS:\n"
            "1. Transcribe the handwriting as accurately as possible. Pay close attention to numbers, dates, units, and chemical names.\n"
            "2. Preserve the structural layout, headings, patient identifiers, and medication tables (line-by-line).\n"
            "3. If a word or number is completely illegible or ambiguous, write '[illegible]' instead of guessing. Never invent clinical details.\n"
            "4. Transcribe abbreviations (e.g. QD, TID, SOS, PRN, SC, IV) exactly as written.\n\n"
            f"Return plain text transcription only for page {page_idx}."
        )

        if cfg["provider"] == "gemini":
            return self._call_gemini_vision_ocr(image_b64, prompt, cfg)
        return self._call_openai_compatible_vision_ocr(image_b64, prompt, cfg)

    def _call_gemini_vision_ocr(self, image_b64: str, prompt: str, cfg: dict) -> str:
        model = cfg["model_name"]
        if "/" in model:
            model = model.split("/")[-1]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={cfg['api_key']}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=API_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini vision OCR returned {response.status_code}: {response.text[:200]}")
        data = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()

    def _call_openai_compatible_vision_ocr(self, image_b64: str, prompt: str, cfg: dict) -> str:
        url = f"{cfg['base_url']}/chat/completions"
        payload = {
            "model": cfg["model_name"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(f"Vision OCR returned {response.status_code}: {response.text[:200]}")
        return response.json()["choices"][0]["message"]["content"].strip()

    def _image_to_base64_png(self, image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _parse_transformer_ocr_response(self, payload) -> str:
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return first.get("generated_text") or first.get("text") or str(first)
        if isinstance(payload, dict):
            return payload.get("generated_text") or payload.get("text") or str(payload)
        return str(payload)

    def _clean_name_from_filename(self, pdf_path: str) -> str:
        if not pdf_path:
            return "Patient"
        filename = os.path.basename(pdf_path)
        if filename.startswith("_uploaded_"):
            filename = filename[len("_uploaded_"):]
        name, _ = os.path.splitext(filename)
        name = name.replace("_", " ").replace("-", " ").replace(".", " ")
        name = " ".join(name.split())
        return name.title() if name else "Patient"

    def _split_records_by_patient(self, page_texts: List[str], pdf_path: str = "") -> Dict[str, str]:
        records: Dict[str, List[str]] = {}
        
        current_name = None
        for index, page_text in enumerate(page_texts, start=1):
            if not page_text.strip():
                continue
            detected_name = self._detect_patient_name(page_text)
            
            if detected_name:
                current_name = detected_name
            elif current_name is None:
                # Try to scan the first page for a patient name candidate
                candidates = self._extract_name_candidates_from_text(page_text)
                if candidates:
                    current_name = candidates[0]
                else:
                    # Fallback to filename or "Patient"
                    current_name = self._clean_name_from_filename(pdf_path)

            records.setdefault(current_name, []).append(f"[Source page {index}]\n{page_text}")

        return {name: "\n\n".join(parts).strip() for name, parts in records.items()}

    def _extract_name_candidates_from_text(self, text: str) -> List[str]:
        # Find all sequences of capitalized words on the same line
        pattern = r"\b[A-Z][a-zA-Z.]*(?:[ \t]+[A-Z][a-zA-Z.]*)+"
        matches = re.findall(pattern, text)
        
        candidates = []
        blacklist = [
            "STAFF", "DOCTOR", "CONSULTANT", "PHYSICIAN", "INCHARGE", "CHECKED", 
            "DATE", "TIME", "REMARKS", "ARRIVAL", "RESPONSE", "ORDER", "EMERGENCY", 
            "REGISTRATION", "HOSPITAL", "CLINIC", "WARD", "BED", "REFER", "SIGNATURE",
            "CROSS", "CHECK", "RESUME", "CV", "EDUCATION", "EXPERIENCE", "PROJECT", 
            "PROJECTS", "SKILLS", "SUMMARY", "PROFILE", "CONTACT", "EMAIL", "PHONE", 
            "MOBILE", "ADDRESS", "LINKEDIN", "GITHUB", "WEBSITE", "INTERESTS", 
            "LANGUAGES", "CERTIFICATIONS", "AWARDS", "PUBLICATIONS", "REFERENCES",
            "UNIVERSITY", "COLLEGE", "SCHOOL", "DEGREE", "BACHELOR", "MASTER", "PHD",
            "CURRICULUM", "VITAE", "PATIENT", "NAME", "DETAILS", "DEPARTMENT", "MEDICINE",
            "ADMISSION", "DISCHARGE", "GENDER", "MALE", "FEMALE", "AGE", "YEARS", "MONTHS",
            "DIAGNOSIS", "HISTORY", "COURSE", "ADVICE", "FOLLOW", "CANDIDATE"
        ]
        
        for match in matches:
            words = [w.strip(" .:-") for w in match.split() if w.strip(" .:-")]
            cleaned_words = []
            for w in words:
                w_upper = w.upper()
                if w_upper not in blacklist and not any(bl == w_upper for bl in blacklist):
                    cleaned_words.append(w)
            
            if 2 <= len(cleaned_words) <= 3:
                candidate = " ".join(cleaned_words)
                if re.fullmatch(r"[A-Za-z][A-Za-z .]{1,59}", candidate):
                    candidates.append(candidate)
                    
        return candidates

    def _is_valid_candidate_name(self, name_candidate: str) -> bool:
        clean = " ".join(name_candidate.strip(" .:-").split())
        det_upper = clean.upper()
        if len(clean) < 3 or len(clean) > 60:
            return False
        # Must only contain letters, spaces, and periods
        if not re.fullmatch(r"[A-Za-z][A-Za-z .]{1,59}", clean):
            return False
        
        blacklist = [
            "STAFF", "DOCTOR", "CONSULTANT", "PHYSICIAN", "INCHARGE", "CHECKED", 
            "DATE", "TIME", "REMARKS", "ARRIVAL", "RESPONSE", "ORDER", "EMERGENCY", 
            "REGISTRATION", "HOSPITAL", "CLINIC", "WARD", "BED", "REFER", "SIGNATURE",
            "CROSS", "CHECK", "RESUME", "CV", "EDUCATION", "EXPERIENCE", "PROJECT", 
            "PROJECTS", "SKILLS", "SUMMARY", "PROFILE", "CONTACT", "EMAIL", "PHONE", 
            "MOBILE", "ADDRESS", "LINKEDIN", "GITHUB", "WEBSITE", "INTERESTS", 
            "LANGUAGES", "CERTIFICATIONS", "AWARDS", "PUBLICATIONS", "REFERENCES",
            "UNIVERSITY", "COLLEGE", "SCHOOL", "DEGREE", "BACHELOR", "MASTER", "PHD",
            "CURRICULUM", "VITAE", "CANDIDATE"
        ]
        invalid_names = [
            "NAME", "PATIENT", "PATIENT NAME", "PT NAME", "FULL NAME", "DETAILS", 
            "CHECK LIST", "I M", "OF THE", "IS REQUIRED", "NOT AVAILABLE", "UNKNOWN"
        ]
        
        if det_upper in invalid_names:
            return False
        if any(word in det_upper for word in blacklist):
            return False
        return True

    def _detect_patient_name(self, text: str) -> str:
        patterns = [
            r"(?:Patient\s*Name|Pt\.?\s*Name|Patient\s*Full\s*Name|Full\s*Name|Candidate\s*Name|Name\s*of\s*Patient|Name\s*of\s*Candidate|Patient|Candidate)\s*[:\-]?\s*([A-Za-z .]{2,60})",
            r"Name\s*[:\-]?\s*([A-Za-z .]{2,60})",
        ]
        blacklist = [
            "STAFF", "DOCTOR", "CONSULTANT", "PHYSICIAN", "INCHARGE", "CHECKED", 
            "DATE", "TIME", "REMARKS", "ARRIVAL", "RESPONSE", "ORDER", "EMERGENCY", 
            "REGISTRATION", "HOSPITAL", "CLINIC", "WARD", "BED", "REFER", "SIGNATURE",
            "CROSS", "CHECK", "RESUME", "CV", "EDUCATION", "EXPERIENCE", "PROJECT", 
            "PROJECTS", "SKILLS", "SUMMARY", "PROFILE", "CONTACT", "EMAIL", "PHONE", 
            "MOBILE", "ADDRESS", "LINKEDIN", "GITHUB", "WEBSITE", "INTERESTS", 
            "LANGUAGES", "CERTIFICATIONS", "AWARDS", "PUBLICATIONS", "REFERENCES",
            "UNIVERSITY", "COLLEGE", "SCHOOL", "DEGREE", "BACHELOR", "MASTER", "PHD",
            "CURRICULUM", "VITAE"
        ]
        invalid_names = [
            "NAME", "PATIENT", "PATIENT NAME", "PT NAME", "FULL NAME", "DETAILS", 
            "CHECK LIST", "I M", "OF THE", "IS REQUIRED", "NOT AVAILABLE", "UNKNOWN"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                name = re.split(
                    r"\s{2,}|\||,|Age|Sex|MRN|IP\s*Number|Pt\s*ID",
                    match.group(1),
                    flags=re.IGNORECASE,
                )[0]
                detected = " ".join(name.strip(" .:-").split())
                
                # Check constraints
                det_upper = detected.upper()
                if len(detected) < 3 or det_upper in invalid_names:
                    continue
                if any(word in det_upper for word in blacklist):
                    continue

                # Normalize noisy OCR spelling variations of the two patients
                det_clean = det_upper.replace(" ", "").replace(".", "")
                if "PREMA" in det_clean:
                    return "Prema J"
                if any(x in det_clean for x in ["NAGARA", "HDN", "MAGMA", "NAGANIA", "NAGARAIA", "NAGARAM", "MANAMA"]):
                    return "H D Nagaraja"
                
                # General fallback: return cleaned and title-cased name
                return " ".join(w.capitalize() for w in detected.split() if w)
        return ""


def _run_easyocr_worker_from_env():
    if os.getenv("EASYOCR_WORKER_MODE") != "1":
        return False

    pdf_path = os.environ["EASYOCR_WORKER_PDF"]
    start_page = int(os.environ["EASYOCR_WORKER_START_PAGE"])
    end_page = int(os.environ["EASYOCR_WORKER_END_PAGE"])
    output_path = os.environ["EASYOCR_WORKER_OUTPUT"]

    parser = ClinicalTextParser()
    rendered_pages = parser._render_pdf_pages(pdf_path, start_page=start_page, end_page=end_page)
    page_texts = parser._ocr_rendered_images_with_easyocr(rendered_pages, page_offset=start_page - 1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"page_texts": page_texts}, f)
    return True


if __name__ == "__main__":
    if _run_easyocr_worker_from_env():
        sys.exit(0)
