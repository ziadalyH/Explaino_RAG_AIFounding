# PDF Documents Directory

Place your PDF documents in this directory.

## File Format

- Standard PDF files (.pdf extension)
- Must contain extractable text (not scanned images without OCR)
- English text content recommended

## Indexing

The system will automatically discover and index all PDF files in this directory when you run:

```bash
python main.py index
```

Or when using Docker with `AUTO_INDEX_ON_STARTUP=true`.

## Processing

PDFs are processed as follows:

1. Text is extracted page by page
2. Each page is split into paragraphs
3. Paragraphs are chunked and embedded
4. Metadata includes: filename, page number, paragraph index
