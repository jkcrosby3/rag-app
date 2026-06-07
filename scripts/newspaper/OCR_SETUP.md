# OCR Improvement Setup Guide

This guide helps you set up modern OCR tools to improve historical newspaper text quality.

## Why Improve OCR?

The original OCR from Library of Congress newspapers is often poor due to:
- 18th century printing quality (faded ink, worn type)
- Old fonts (Gothic/Old English)
- Historical spelling variations
- Multi-column layouts
- Paper damage

Modern OCR tools (Tesseract 5.x) with proper preprocessing can significantly improve results.

## Installation Steps

### 1. Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (tesseract-ocr-w64-setup-5.x.x.exe)
3. During installation, note the installation path (default: `C:\Program Files\Tesseract-OCR`)
4. Add to PATH or update `TESSERACT_PATH` in `improve_ocr.py`

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### 2. Install Python Dependencies

```bash
pip install -r scripts/newspaper/requirements_ocr.txt
```

This installs:
- `pytesseract` - Python wrapper for Tesseract
- `Pillow` - Image processing
- `requests` - Download images from URLs
- `pdf2image` - PDF support (optional)

### 3. Verify Installation

```bash
tesseract --version
```

Should output something like:
```
tesseract 5.3.0
```

### 4. Configure Path (Windows only)

If Tesseract is not in your PATH, edit `scripts/newspaper/improve_ocr.py`:

```python
# Line ~30
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Usage

### Test Run (5 newspapers)

```bash
python scripts/newspaper/improve_ocr.py --limit 5 --compare
```

This will:
1. Download 5 newspaper images from Library of Congress
2. Preprocess images (contrast, resize)
3. Run modern OCR
4. Compare with original OCR
5. Save improved text as `*_improved.txt`

### Process All Newspapers

```bash
python scripts/newspaper/improve_ocr.py --compare
```

**Warning**: This processes all 65 newspapers. Each takes 1-2 minutes = ~2 hours total.

### Process Specific Newspaper

```bash
python scripts/newspaper/improve_ocr.py --lccn sn85025609 --compare
```

## Expected Results

**Original OCR Issues:**
```
St .S:oijr.!s^-v«.nment,a, »b« Engi fl, *L or Shift
in 1777. iocnu'ner tiling tl*. JiiUitns at Trenton, 1 wat aiatica y
```

**Improved OCR:**
```
St. George's government, as the English, or Shift
in 1777. encountered thing the Indians at Trenton, I was attacked
```

Not perfect, but significantly better for:
- Name extraction
- Keyword matching
- Cross-referencing with pension files

## Workflow

1. **Test first**: Run with `--limit 5 --compare` to verify quality improvement
2. **Review results**: Check `*_improved.txt` files
3. **If better**: Process all newspapers
4. **Re-enrich**: Run `enrich_newspaper_files.py` on improved text
5. **Cross-reference**: Run `cross_reference_veterans.py` with better names

## Troubleshooting

**Error: "Tesseract not found"**
- Verify installation: `tesseract --version`
- Windows: Add to PATH or set `TESSERACT_PATH` in script
- Mac/Linux: Reinstall with package manager

**Error: "No module named 'pytesseract'"**
- Install dependencies: `pip install -r scripts/newspaper/requirements_ocr.txt`

**Error: "Download failed"**
- Check internet connection
- Library of Congress URLs may be temporarily unavailable
- Try again later

**OCR produces gibberish**
- Some newspapers are too damaged for any OCR
- Check original image quality at Library of Congress URL
- Skip these files

## Performance Notes

- **CPU-intensive**: Uses all available CPU cores
- **Time**: 1-2 minutes per newspaper
- **Memory**: ~500MB per newspaper
- **Disk**: ~50KB per improved text file
- **Total time**: ~2 hours for all 65 newspapers

## Alternative: Cloud OCR

If local processing is too slow, consider:
- **Google Cloud Vision API** (paid, very accurate)
- **AWS Textract** (paid, good for historical documents)
- **Azure Computer Vision** (paid, supports old fonts)

These require API keys and have costs, but may produce better results for heavily damaged newspapers.

## Next Steps

After improving OCR:
1. Compare extraction quality (names, places, battles)
2. Re-run enrichment on improved text
3. Cross-reference veterans with better name matching
4. Document which newspapers benefited most from re-OCR
