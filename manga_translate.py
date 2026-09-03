import zipfile
import os
from pathlib import Path
from dataclasses import dataclass
import re
import shutil
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import time


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

_bubble_model_ = None

def get_bubble_model():
    global _bubble_model_
    if _bubble_model_ is None:
        weights_path = hf_hub_download(
            repo_id="ogkalu/comic-speech-bubble-detector-yolov8m",
            filename="comic-speech-bubble-detector.pt",
        )
        _bubble_model_ = YOLO(weights_path)

    return _bubble_model_

def detect_bubbles(image_path: Path, conf: float = 0.35, min_width: int = 40, min_height: int = 30, debug: bool = False,) -> list[tuple[int, int, int, int]]:
    model = get_bubble_model()
    results = model.predict(str(image_path), conf = conf, verbose = False)[0]

    bubbles = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        if w < min_width or h < min_height:
            if debug:
                print(f"skipped tiny box {(x, y, w, h)} conf = {float(box.conf[0]):.2f}")
            continue

        bubbles.append((x, y, w, h))

        if debug:
            print(f"bubble {(x, y, w, h)} conf = {float(box.conf[0]):.2f}")

    bubbles.sort(key = lambda b: (b[1] // 150, -b[0]))

    return bubbles

def crop_bubbles(image_path: Path, bubbles: list[tuple[int, int, int, int]]) -> list[Path]:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    cropped_paths = []
    for (x, y, w, h) in bubbles:
        cropped_img = img.crop((x, y, x + w, y + h))
        cropped_paths.append(cropped_img)

    return cropped_paths

_CJK_PATTERN_ = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9f]')

def contains_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN_.search(text))

def needs_cjk_font(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)

_JP_PUNCT_MAP = {
    "\u2025": "...",
    "\u2026": "...",
    "\u22ef": "...",
    "\u30fb": "",
    "\u301c": "~",
    "\uff5e": "~",
    "\u3002": ".",
    "\u3001": ",",
}

def normalize_punctuation(text: str) -> str:
    for jp_char, ascii_equiv in _JP_PUNCT_MAP.items():
        text = text.replace(jp_char, ascii_equiv)

    return text

def is_meaningful_text(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False
    return bool(re.search(r'\w', stripped, re.UNICODE))

def get_font(size: int, cjk: bool = False):
    from PIL import ImageFont

    if cjk:
        candidates = [
            "C:\\Windows\\Fonts\\msgothic.ttc",
            "C:\\Windows\\Fonts\\meiryo.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
    else:
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

def wrap_text_to_width(draw, text: str, font, max_width: int) -> tuple[list[str], bool]:
    words = text.split()

    if not words:
        return [], False

    forced_split = False

    def split_long_word(word: str) -> list[str]:
        chunks, current = [], ""

        for ch in word:
            test = current + ch
            bbox = draw.textbbox((0, 0), test, font = font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = test
            else:
                chunks.append(current)
                current = ch

        if current:
            chunks.append(current)

        return chunks
    
    lines, current_line = [], ""

    for word in words:
        candidate = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font = font)

        if bbox[2] - bbox[0] <= max_width:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
                current_line = ""

            word_bbox = draw.textbbox((0, 0), word, font = font)

            if word_bbox[2] - word_bbox[0] > max_width:
                pieces = split_long_word(word)
                lines.extend(pieces[:-1])
                current_line = pieces[-1] if pieces else ""
                forced_split = True
            else:
                current_line = word

    if current_line:
        lines.append(current_line)

    return lines, forced_split

@dataclass
class BubbleLayout:
    fill_box: list
    corner_radius: int
    lines: list[str]
    font: object
    center_x: int
    start_y: int
    text_color: tuple
    bg_color: tuple

def computer_bubble_layout(draw, box: tuple[int, int, int, int], text: str, bg_color = (255, 255, 255), text_color = (0, 0, 0), max_font_size: int = 20, min_font_size: int = 6) -> BubbleLayout:
    x, y, w, h = box
    padding = 6
    pad_out = 3

    fill_box = [x - pad_out, y - pad_out, x + w + pad_out, y + h + pad_out]
    corner_radius = max(min(w, h) // 6, 4)

    if not text.strip():
        return BubbleLayout(fill_box, corner_radius, [], None, x + w // 2, y, text_color, bg_color)

    cjk = needs_cjk_font(text)
    max_width = max(w - padding * 2, 10)
    max_height = max(h - padding * 2, 10)

    font_size = max(min(h // 5, max_font_size), min_font_size)

    best_fit = None
    fallback_fit = None

    while font_size >= min_font_size:
        font = get_font(font_size, cjk = cjk)
        lines, forced_split = wrap_text_to_width(draw, text, font, max_width)

        line_heights = [draw.textbbox((0, 0), line, font = font)[3] - draw.textbbox((0, 0), line, font = font)[1] for line in lines]
        total_height = sum(line_heights) + (len(lines) - 1) * 2

        if total_height <= max_height:
            if not forced_split:
                best_fit = (lines, font)
                break
            elif fallback_fit is None:
                fallback_fit = (lines, font)

        font_size -= 2

    if best_fit:
        lines, font = best_fit
    elif fallback_fit:
        lines, font = fallback_fit
    else:
        font = get_font(min_font_size, cjk = cjk)
        lines, _ = wrap_text_to_width(draw, text, font, max_width)

    word_widths = [draw.textbbox((0, 0), word, font = font)[2] for word in text.split()]
    widest_word = max(word_widths, default = 0)
    used_width = max(max_width, widest_word)

    if used_width > max_width:
        lines, _  = wrap_text_to_width(draw, text, font, used_width)
        grow = used_width - max_width
        fill_box[0] -= grow // 2
        fill_box[2] += grow - grow // 2

    line_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines]
    total_height = sum(line_heights) + (len(lines) - 1) * 2

    center_x = (fill_box[0] + fill_box[2]) // 2
    start_y = y + (h - total_height) // 2

    return BubbleLayout(fill_box, corner_radius, lines, font, center_x, start_y,text_color, bg_color)

def draw_bubble_background(draw, layout: BubbleLayout):
    draw.rounded_rectangle(layout.fill_box, radius = layout.corner_radius, fill = layout.bg_color)

def draw_bubble_text(draw, layout: BubbleLayout):
    if not layout.lines:
        return

    current_y = layout.start_y

    for line in layout.lines:
        bbox = draw.textbbox((0, 0), line, font = layout.font)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]
        line_x = layout.center_x - line_w // 2
        draw.text((line_x, current_y), line, font=layout.font, fill=layout.text_color)
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

def call_translator_api(text: str, retries: int = 10, delay: float = 3) -> str:
    from deep_translator import GoogleTranslator

    if not text or not text.strip():
        return ""

    last_error = None
    for attempt in range(retries):
        try:
            translated = GoogleTranslator(source = 'ja', target = 'en').translate(text)
            return normalize_punctuation(translated)
        
        except Exception as e:
            last_error = e
            time.sleep(delay)
    
    print(f"translation failed after {retries} attempts for '{text}': {last_error}")
    return normalize_punctuation(f"[untranslated: {text}]")

def process_page(pg: MangaPage) -> tuple[int, str]:
    from PIL import Image, ImageDraw

    bubbles = detect_bubbles(pg.path)
    page_img = Image.open(pg.path).convert("RGB")
    draw = ImageDraw.Draw(page_img)

    if not bubbles:
        japanese_text = call_japanese_image_to_text_api(page_img)
        english_text = call_translator_api(japanese_text)
        save_typeset_page(page_img, pg.page_number)
        return pg.page_number, english_text

    cropped_images = crop_bubbles(pg.path, bubbles)

    lines = []
    layouts: list[BubbleLayout] = []

    for box, cropped_img in zip(bubbles, cropped_images):
        japanese_text = call_japanese_image_to_text_api(cropped_img)

        if not is_meaningful_text(japanese_text):
            continue

        english_text = call_translator_api(japanese_text)
        lines.append(english_text)
        layouts.append(computer_bubble_layout(draw, box, english_text))

    for layout in layouts:
        draw_bubble_background(draw, layout)

    for layout in layouts:
        draw_bubble_text(draw, layout)
        
    save_typeset_page(page_img, pg.page_number)

    page_text = "\n".join(lines)
    return pg.page_number, page_text

def save_typeset_page(page_img, page_number: int):
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    out_path = Path(OUTPUT_IMG_DIR) / f"page_{page_number:03d}.png"
    page_img.save(out_path)

def debug_bubbles(image_path: Path, bubbles: list[tuple[int, int, int, int]], out_path: str = "debug_bubbles.png"):
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in bubbles:
        draw.rectangle([x, y, x + w, y + h], outline = (255, 0, 0), width = 3)
    img.save(out_path)
    
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