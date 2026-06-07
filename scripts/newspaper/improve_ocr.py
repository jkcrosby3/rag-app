"""
Improved OCR Script for Historical Newspapers

Downloads newspaper images and re-processes with modern OCR (Tesseract)
to improve text quality compared to original OCR.

Requirements:
    pip install pytesseract pillow requests pdf2image
    
    Also requires Tesseract OCR installed:
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Set TESSERACT_PATH below to your installation

Usage:
    python scripts/newspaper/improve_ocr.py [--limit N] [--compare]
"""

import json
import requests
from pathlib import Path
from typing import Optional, Dict
import argparse
from io import BytesIO

try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install pytesseract pillow requests pdf2image")
    exit(1)

# Configure Tesseract path (update for your system)
# Windows example: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Mac/Linux: usually just 'tesseract' if in PATH
TESSERACT_PATH = r'C:\Users\639250\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def download_image(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Load image from response
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"    ⚠️  Download failed: {e}")
        return None


def preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess image for better OCR results on historical newspapers."""
    from PIL import ImageEnhance, ImageFilter
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize first if too small (OCR works better on larger images)
    width, height = img.size
    if width < 3000:
        scale = 3000 / width
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Denoise (helps with old paper texture and damage)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Increase sharpness (helps with blurry old fonts)
    sharpener = ImageEnhance.Sharpness(img)
    img = sharpener.enhance(2.0)
    
    # Increase contrast (helps with faded ink)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    
    # Brightness adjustment (helps with yellowed paper)
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.2)
    
    return img


def ocr_image(img: Image.Image) -> str:
    """Perform OCR on image using Tesseract with historical text optimization."""
    try:
        # Enhanced config for historical newspapers with old typography
        # --oem 1: Use LSTM OCR engine only (better for historical fonts)
        # --psm 1: Automatic page segmentation with OSD
        # -c tessedit_char_whitelist: Allow old English characters
        # -c preserve_interword_spaces=1: Better word spacing
        custom_config = r'--oem 1 --psm 1 -c preserve_interword_spaces=1'
        
        text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
        
        # Post-process: Fix common old English typography issues
        text = fix_old_english_typography(text)
        
        return text
    except Exception as e:
        print(f"    ⚠️  OCR failed: {e}")
        return ""


def fix_old_english_typography(text: str) -> str:
    """Fix common old English typography OCR errors (long s, ligatures)."""
    import re
    
    # Common long s (ſ) and ligature misreads
    replacements = {
        'ſ': 's',  # Long s to regular s
        'ﬁ': 'fi',  # fi ligature
        'ﬂ': 'fl',  # fl ligature
        'ﬀ': 'ff',  # ff ligature
        'ſt': 'st',  # Long s + t
        'ſh': 'sh',  # Long s + h
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # AGGRESSIVE: Fix common OCR misreads where long s (ſ) is read as 'f'
    # Order matters - do most specific patterns first
    
    # Pattern 1: "fh" at word start -> "sh" (fhall -> shall, fhould -> should, fhort -> short)
    text = re.sub(r'\bfh', 'sh', text)
    
    # Pattern 2: "fs" anywhere -> "ss" (aflortment -> assortment, paflage -> passage)
    text = re.sub(r'fs', 'ss', text)
    
    # Pattern 3: "ff" between vowels -> "ss" (poffible -> possible, neceflary -> necessary)
    text = re.sub(r'([aeiou])ff([aeiou])', r'\1ss\2', text)
    
    # Pattern 4: "fl" between vowels -> "ss" (poflessions -> possessions, profeflion -> profession)
    text = re.sub(r'([aeiou])fl([aeiou])', r'\1ss\2', text)
    
    # Pattern 5: "f" before vowel+d at word end -> "s" (purchafed -> purchased, raifed -> raised)
    text = re.sub(r'f([aeiou]d\b)', r's\1', text)
    
    # Pattern 6: "f" before vowel+s at word end -> "s" (witneffes -> witnesses, clafs -> class)
    text = re.sub(r'f([aeiou])(s+\b)', r's\1\2', text)
    
    # Pattern 7: "f" before "ion" -> "s" (Seflion -> Session, profeffion -> profession)
    text = re.sub(r'f(ion\b)', r's\1', text)
    
    # Pattern 8: "f" before "t" at word end -> "s" (moft -> most, firft -> first, sirft -> sirst)
    text = re.sub(r'([aeiou])ft\b', r'\1st', text)
    
    # Pattern 9: "f" before "t" after "r" -> "s" (sirft -> sirst -> first)
    text = re.sub(r'([bcdfghjklmnpqrstvwxyz])ft\b', r'\1st', text)
    
    # Pattern 10: "fu" at word start -> "su" (furvey -> survey, fubject -> subject, fuperior -> superior)
    text = re.sub(r'\bfu', 'su', text)
    
    # Pattern 11: "depofit" -> "deposit", "pofition" -> "position"
    text = re.sub(r'([aeiou])f([aeiou]t)', r'\1s\2', text)
    
    # Pattern 12: "f" at word start before vowel -> "s" (fo -> so, fmall -> small, fevere -> severe)
    # BUT: Protect common words that legitimately start with "f"
    # Save legitimate "f" words first
    protected_f_words = {
        'for', 'from', 'first', 'former', 'fortunately', 'fortune', 'four', 'fourteen',
        'follow', 'followed', 'following', 'force', 'forced', 'foreign', 'forest',
        'form', 'formed', 'fort', 'forward', 'found', 'foundation', 'france', 'french'
    }
    
    # Only apply if NOT a protected word
    words = text.split()
    fixed_words = []
    for word in words:
        clean_word = word.lower().strip('.,;:!?"\'')
        if clean_word in protected_f_words:
            fixed_words.append(word)  # Keep as-is
        else:
            # Apply the f->s pattern
            fixed_word = re.sub(r'^f([aeiou])', r's\1', word)
            fixed_words.append(fixed_word)
    text = ' '.join(fixed_words)
    
    # Pattern 13: "f" before consonant in middle of word -> "s" (difcovery -> discovery, affords -> assords)
    text = re.sub(r'([aeiou])f([bcdfghjklmnpqrstvwxyz])', r'\1s\2', text)
    
    # Pattern 14: "f" before "e" at word end -> "s" (arifes -> arises, caf-s -> cases)
    text = re.sub(r'f(e[sd]?\b)', r's\1', text)
    
    # Pattern 15: "ftr" -> "str" (ftreet -> street, ftrong -> strong)
    text = re.sub(r'ftr', 'str', text)
    
    # Pattern 16: Common OCR character confusions
    # Fix common misreads that aren't long s related
    common_fixes = {
        'asit': 'as it',
        'ina': 'in a',
        'sor': 'for',  # Catch any remaining sor->for
        'srom': 'from',  # Catch any remaining srom->from
        'sirst': 'first',  # Fix the sirst->first issue
        'sormer': 'former',  # Fix sormer->former
        'sortunately': 'fortunately',  # Fix sortunately->fortunately
        'diforder': 'disorder',
        'dissiculty': 'difficulty',
        'perse@tly': 'perfectly',
        'fmall': 'small',
        'fleth': 'flesh',
        'asflided': 'afflicted',
        'disserent': 'different',
        'benesit': 'benefit',
        'advertise-Ments': 'advertisements',
        'expeations': 'expectations',
        'informa-tion': 'information',
        'publith': 'publish',
        'assiicicd': 'afflicted',
        'atrend': 'attend',
        'bu-siness': 'business',
        'botties': 'bottles',
        'rectored': 'restored',
        'esf': 'eff',
        'tlixir': 'elixir',
        'suppole': 'suppose',
        'sicac': 'ficac',
        'pleaiure': 'pleasure',
        'defired': 'desired',
        'esse&': 'effect',
        'aitacked': 'attacked',
        'sailed': 'failed',
        'sa': 'so',
    }
    
    for wrong, right in common_fixes.items():
        text = text.replace(wrong, right)
    
    return text


def compare_ocr_quality(original: str, improved: str) -> Dict:
    """Compare original and improved OCR quality."""
    
    def count_words(text):
        return len(text.split())
    
    def count_alpha_ratio(text):
        if not text:
            return 0
        alpha = sum(1 for c in text if c.isalpha())
        return alpha / len(text)
    
    def count_special_chars(text):
        return sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,;:!?-"\'')
    
    original_words = count_words(original)
    improved_words = count_words(improved)
    
    original_alpha = count_alpha_ratio(original)
    improved_alpha = count_alpha_ratio(improved)
    
    original_special = count_special_chars(original)
    improved_special = count_special_chars(improved)
    
    return {
        "original_word_count": original_words,
        "improved_word_count": improved_words,
        "word_count_change": improved_words - original_words,
        "original_alpha_ratio": round(original_alpha, 3),
        "improved_alpha_ratio": round(improved_alpha, 3),
        "original_special_chars": original_special,
        "improved_special_chars": improved_special,
        "quality_improved": improved_alpha > original_alpha and improved_special < original_special
    }


def process_newspaper(metadata_file: Path, compare: bool = False) -> Optional[Dict]:
    """Process a single newspaper file."""
    
    try:
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        lccn = metadata.get("lccn", metadata_file.stem.replace("_metadata", ""))
        print(f"\n📰 Processing: {lccn}")
        print(f"   Title: {metadata.get('newspaper_title', 'Unknown')}")
        print(f"   Date: {metadata.get('issue_date', 'Unknown')}")
        
        # Get image URL (prefer JPEG2000, fallback to thumbnail)
        image_url = metadata.get("jpeg2000_url") or metadata.get("thumbnail_url")
        
        if not image_url:
            print("   ⚠️  No image URL found")
            return None
        
        print(f"   📥 Downloading image...")
        img = download_image(image_url)
        
        if not img:
            return None
        
        print(f"   🖼️  Image size: {img.size[0]}x{img.size[1]}")
        print(f"   🔧 Preprocessing...")
        img = preprocess_image(img)
        
        print(f"   📝 Running OCR (this may take a minute)...")
        improved_text = ocr_image(img)
        
        if not improved_text:
            print("   ❌ OCR produced no text")
            return None
        
        print(f"   ✓ OCR complete: {len(improved_text)} characters")
        
        # Save improved OCR
        text_file = metadata_file.parent / f"{lccn}_improved.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"Document ID: {lccn}\n")
            f.write(f"Improved OCR (Tesseract)\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(improved_text)
        
        print(f"   💾 Saved to: {text_file.name}")
        
        # Compare if requested
        comparison = None
        if compare:
            original_text = metadata.get("ocr_text", "")
            if original_text:
                comparison = compare_ocr_quality(original_text, improved_text)
                print(f"\n   📊 Quality Comparison:")
                print(f"      Original words: {comparison['original_word_count']}")
                print(f"      Improved words: {comparison['improved_word_count']}")
                print(f"      Change: {comparison['word_count_change']:+d}")
                print(f"      Quality improved: {'✓ Yes' if comparison['quality_improved'] else '✗ No'}")
        
        return {
            "lccn": lccn,
            "improved_text_file": str(text_file),
            "improved_char_count": len(improved_text),
            "comparison": comparison
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Improve newspaper OCR quality")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--compare", action="store_true", help="Compare with original OCR")
    parser.add_argument("--lccn", type=str, help="Process specific LCCN only")
    args = parser.parse_args()
    
    # Path to newspaper metadata
    base_dir = Path(__file__).parent.parent.parent / "data" / "documents" / "smithsonian" / "newspapers" / "text_files"
    
    print("\n" + "=" * 70)
    print("🔍 IMPROVED OCR FOR HISTORICAL NEWSPAPERS")
    print("=" * 70)
    
    # Check Tesseract installation
    try:
        version = pytesseract.get_tesseract_version()
        print(f"\n✓ Tesseract version: {version}")
    except Exception as e:
        print(f"\n❌ Tesseract not found: {e}")
        print("\nPlease install Tesseract OCR:")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Mac: brew install tesseract")
        print("  Linux: sudo apt-get install tesseract-ocr")
        return
    
    # Find metadata files
    if args.lccn:
        metadata_files = [base_dir / f"{args.lccn}_metadata.json"]
    else:
        metadata_files = list(base_dir.glob("*_metadata.json"))
    
    if args.limit:
        metadata_files = metadata_files[:args.limit]
    
    print(f"\n📂 Found {len(metadata_files)} metadata files")
    
    if args.limit:
        print(f"⚠️  Processing limited to {args.limit} files")
    
    # Process files
    results = []
    processed = 0
    errors = 0
    
    for metadata_file in metadata_files:
        result = process_newspaper(metadata_file, compare=args.compare)
        
        if result:
            results.append(result)
            processed += 1
        else:
            errors += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\nTotal files: {len(metadata_files)}")
    print(f"Successfully processed: {processed}")
    print(f"Errors: {errors}")
    
    if args.compare and results:
        improved_count = sum(1 for r in results if r.get("comparison", {}).get("quality_improved"))
        print(f"\nQuality improved: {improved_count}/{len(results)} files")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
