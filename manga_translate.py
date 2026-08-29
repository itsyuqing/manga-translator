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
OUTPUT_IMG_DIR = "translated_pages"

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

def get_font(size: int):
    from PIL import ImageFont

    candidates = [
        "arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()

def wrap_text_to_width(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line}, word"
        bbox = draw.textbox((0, 0), text_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines

def typeset_bubble(draw, box: tuple[int, int, int, int], text: str, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    x, y, w, h = box
    padding = 6

    draw.rectangle([x, y, x + w, y + h], fill=bg_color)

    if not text.strip():
        return

    max_width = max(w - padding * 2, 10)
    max_height = max(h - padding * 2, 10)

    font_size = max(min(h // 4, 28), 8)

    while font_size >= 8:
        font = get_font(font_size)
        lines = wrap_text_to_width(draw, text, font, max_width)

        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font = font)
            line_heights.append(bbox[3] - bbox[1])
        total_height = sum(line_heights) + (len(lines) - 1) * 2

        if total_height <= max_height:
            break
        font_size -= 2
    else:
        font = get_font(8)
        lines = wrap_text_to_width(draw, text, font, max_width)

        total_height = sum(
            draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
            for line in lines
        ) + (len(lines) - 1) * 2

        current_y = y + (h - total_height) // 2

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
            line_x = x + (w - line_w) // 2
            draw.text((line_x, current_y), line, font=font, fill=text_color)
            current_y += line_h + 2

        
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
    from PIL import Image, ImageDraw

    bubbles = detect_bubbles(pg.path)
    page_img = Image.open(pg.path).convert("RBG")
    draw = ImageDraw.Draw(page_img)

    if not bubbles:
        from PIL import Image
        japanese_text = call_japanese_image_to_text_api(Image.open(pg.path))
        english_text = call_translator_api(japanese_text)
        return pg.page_number, english_text

    cropped_paths = crop_bubbles(pg.path, bubbles)

    lines = []
    for box, cropped_img in zip(bubbles, cropped_images):
        japanese_text = call_japanese_image_to_text_api(cropped_img)
        if not japanese_text.strip():
            continue
        english_text = call_translator_api(japanese_text)
        lines.append(english_text)

        typeset_bubble(draw, box, english_text)

    save_typeset_page(page_img, pg.page_number)

    page_text = "\n".join(lines)
    return pg.page_number, page_text

def save_typeset_page(page_img, page_number: int):
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    out_path = Path(OUTPUT_IMG_DIR) / f"page_{page_number:03d}.png"
    page_img.save(out_path)
    
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