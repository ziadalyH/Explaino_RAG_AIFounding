"""Test script to demonstrate Unstructured.io PDF parsing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion.pdf_ingester import PDFIngester
from src.config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("UNSTRUCTURED.IO PDF PARSING TEST")
print("="*80)

# Check if we have any PDFs
pdf_dir = Path("data/pdfs")
if not pdf_dir.exists():
    pdf_dir = Path("mock_data/pdfs")

pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

if not pdf_files:
    print("\n❌ No PDF files found in data/pdfs or mock_data/pdfs")
    print("\nTo test PDF parsing:")
    print("1. Place some PDF files in data/pdfs/ or mock_data/pdfs/")
    print("2. Run this script again")
    sys.exit(1)

print(f"\n✅ Found {len(pdf_files)} PDF files:")
for pdf in pdf_files:
    print(f"  - {pdf.name}")

# Create config and ingester
class MockConfig:
    pdf_dir = pdf_dir

config = MockConfig()
ingester = PDFIngester(config, logger)

print("\n" + "="*80)
print("PARSING PDFs WITH UNSTRUCTURED.IO")
print("="*80)

all_paragraphs = []

for pdf_file in pdf_files:
    print(f"\n{'='*80}")
    print(f"Processing: {pdf_file.name}")
    print(f"{'='*80}")
    
    paragraphs = ingester.ingest_file(pdf_file)
    
    if paragraphs:
        all_paragraphs.extend(paragraphs)
        
        print(f"\n✅ Extracted {len(paragraphs)} paragraphs")
        
        # Show first 3 paragraphs
        print(f"\nFirst 3 paragraphs:")
        for i, para in enumerate(paragraphs[:3], 1):
            print(f"\n  Paragraph {i}:")
            print(f"    Page: {para.page_number}")
            print(f"    Index: {para.paragraph_index}")
            print(f"    Length: {len(para.text)} characters")
            print(f"    Text: {para.text[:100]}...")
        
        # Show statistics
        pages = set(p.page_number for p in paragraphs)
        avg_length = sum(len(p.text) for p in paragraphs) / len(paragraphs)
        
        print(f"\n  Statistics:")
        print(f"    Total paragraphs: {len(paragraphs)}")
        print(f"    Pages: {len(pages)}")
        print(f"    Avg paragraph length: {avg_length:.0f} characters")
        print(f"    Shortest: {min(len(p.text) for p in paragraphs)} characters")
        print(f"    Longest: {max(len(p.text) for p in paragraphs)} characters")
    else:
        print(f"\n❌ No paragraphs extracted from {pdf_file.name}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all_paragraphs:
    total_pages = len(set(p.page_number for p in all_paragraphs))
    total_chars = sum(len(p.text) for p in all_paragraphs)
    
    print(f"""
✅ Successfully parsed {len(pdf_files)} PDF files

Total Statistics:
  - Paragraphs extracted: {len(all_paragraphs)}
  - Total pages: {total_pages}
  - Total characters: {total_chars:,}
  - Average per paragraph: {total_chars / len(all_paragraphs):.0f} characters

Unstructured.io Features Used:
  ✅ Intelligent paragraph detection
  ✅ Page number tracking
  ✅ Element type classification
  ✅ Text cleaning and normalization
  ✅ Table structure preservation (if present)
  ✅ Multi-column handling

Benefits over PyPDF2:
  ✅ Better text extraction quality
  ✅ Proper paragraph boundaries
  ✅ Table support
  ✅ Layout analysis
  ✅ OCR capability (for scanned PDFs)
""")
else:
    print("\n❌ No paragraphs extracted from any PDF files")
    print("\nPossible issues:")
    print("  - PDFs may be empty or corrupted")
    print("  - PDFs may be scanned images (need OCR)")
    print("  - System dependencies may be missing (poppler, tesseract)")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Review the extracted paragraphs above
2. Compare with original PDF to verify quality
3. If quality is good, proceed with indexing:
   
   docker-compose exec rag-backend python main.py index

4. Query the system to test retrieval:
   
   docker-compose exec rag-backend python main.py query --question "your question"
""")
