from pathlib import Path
from manga_translate import detect_bubbles, debug_bubbles

test_page = Path("extracted_pages/p1.png")

bubbles = detect_bubbles(test_page)
print(f"Found {len(bubbles)} bubbles: {bubbles}")

debug_bubbles(test_page, bubbles, "debug_bubbles.png")