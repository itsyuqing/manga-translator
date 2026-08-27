import zipfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import shutil
import cv2


ZIP_Path = "manga raws.zip"
EXTRACT_DIR = "extracted_pages"
OUTPUT_FILE = "translated_manga.txt"
MAX_WORKERS = 4 #idk

@dataclass
class MangaPage:
    page_number: int
    path: Path

def unzip_folder(zip_path: str, extract_dir: str) -> list[MangaPage]:
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    def extract_number(path: Path) -> int:
        match = re.search(r'\d+', path.stem)
        return int(match.group()) if match else 0

    png_files = sorted(Path(extract_dir).glob('*.png'), key=extract_number)

    pages = [
        MangaPage(page_number=i, path=p)
        for i, p in enumerate(png_files, start=1)
    ]

    return pages

def detect_bubbles(image_path: Path, min_area: int = 2000) -> list[tuple[int, int, int, int]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    height, width = img.shape

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        if w > width * 0.9 and h > height * 0.9:
            continue

        aspect = w / h if h else 0
        if aspect < 0.5 or aspect > 5:
            continue

        filled_ratio = area / (w * h) if (w * h) else 0
        if filled_ratio < 0.3:
            continue

        bubbles.append((x, y, w, h))

    bubbles.sort(key=lambda b: (b[1] // 150, -b[0]))

    return bubbles

def crop_bubbles(image_path: Path, bubbles: list[tuple[int, int, int, int]]) -> list[Path]:
    from PIL import Image

    img = Image.open(image_path)
    cropped_paths = []
    for (x, y, w, h) in bubbles:
        cropped_img = img.crop((x, y, x + w, y + h))
        cropped_paths.append(cropped_img)

    return cropped_paths

_mocr = None

def get_mocr():
    global _mocr
    if _mocr is None:
        from manga_ocr import MangaOcr
        _mocr = MangaOcr()
    return _mocr

def call_japanese_image_to_text_api(image) -> str:
    mocr = get_mocr()
    return mocr(image)

def call_translator_api(text: str) -> str: #idk how right this is
    from googletrans import Translator
    translator = Translator()
    translated = translator.translate(text, src='ja', dest='en')
    return translated.text

def process_page(pg: MangaPage) -> tuple[int, str]:
    bubbles = detect_bubbles(pg.path)

    if not bubbles:
        from PIL import Image
        japanese_text = call_japanese_image_to_text_api(Image.open(pg.path))
        english_text = call_translator_api(japanese_text)
        return pg.page_number, english_text

    cropped_paths = crop_bubbles(pg.path, bubbles)

    lines = []
    for cropped_path in cropped_paths:
        japanese_text = call_japanese_image_to_text_api(cropped_path)
        if not japanese_text.strip():
            continue
        english_text = call_translator_api(japanese_text)
        lines.append(english_text)

    page_text = "\n".join(lines)
    return pg.page_number, page_text

def main():
    pages = unzip_folder(ZIP_Path, EXTRACT_DIR)

    get_mocr()

    results: dict[int, str] = {}

    for pg in pages:
        try:
            page_number, english_text = process_page(pg)
            results[page_number] = english_text
            print(f"Page {page_number} done.")
        except Exception as e:
            print(f"Error processing page {pg.page_number}: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for page_number in sorted(results.keys()):
            f.write(f"Page {page_number}:\n{results[page_number]}\n\n")


if __name__ == "__main__":
    main()