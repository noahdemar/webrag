# Webrag (we brag)

Private, in-browser search for PDFs and scanned images. Drop files in and search them by exact words/regex or fuzzily (optionally by meaning too), then jump to the source page with the match highlighted on the original document. Nothing is uploaded. Parsing, OCR, indexing and search all run on your machine.

## Features

- **Text extraction with OCR fallback.** Uses the PDF text layer when it's trustworthy and falls back to on-device OCR (ONNX models in `models/`) when it isn't. Lines are put into reading order by a small layout model.
- **Exact / regex search on WebGPU.** A table-driven NFA kernel runs one GPU thread per line. It mirrors a JS implementation that's used when WebGPU isn't available.
- **Fuzzy search on WebGPU.** Bit-parallel approximate matching (Wu–Manber bitap, up to 3 edits) with an anchored backward pass to recover where each match starts. It ignores case, spaces and hyphens, so `important` finds the OCR-split `im portant` and `villagisation` finds `Villagization`. About 5–15 ms over 18.7k lines, against 70–370 ms for the JS fallback.
- **Semantic search (opt-in, under Advanced).** Off by default. Nothing is shown, downloaded or run until you switch it on. [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) (q4 ONNX) runs on WebGPU via onnxruntime-web. Pages are split into passages of about 80 words, embedded at 256 dimensions (Matryoshka truncation) and kept in IndexedDB. The model (≈200 MB) downloads once from Hugging Face the first time you turn semantic search on, and is then cached by the browser.
- **Highlights on the source page.** The viewer re-reads the page's text runs, rebuilds the matched line, and measures glyph advances in the PDF's own font. This gives word-level boxes on the rendered original, including scanned books with an invisible OCR text layer. Pages that only have OCR text fall back to estimated positions inside the line box.
- **Page map.** A grid of every page, coloured where results land. Click a page to open it, use ← / → to step between result pages, and press Enter in the search box to jump to the next one.
- **Library and exports.** Everything persists in the browser: extracted text in IndexedDB, original files in OPFS so previews survive reloads. Export a `.jsonl` dump (re-importable) or a **Parquet** file with one row per line (`file, page, line, text, x0, y0, x1, y1`; boxes in PDF points, top-left origin), ready for DuckDB, pandas or Polars.

## Run it

It's a static site, but multithreaded WASM needs cross-origin isolation headers, so use the bundled server:

```bash
python3 serve.py 8000
```

Then open <http://localhost:8000>. You'll want a browser with WebGPU (recent Chrome, Edge or Safari) for the fast paths. Everything also works without it, just more slowly.

Keyboard: `/` focuses search, `Enter` opens the next result page, `←` / `→` step through result pages, `Esc` closes the viewer.

## Layout

```
index.html, styles.css    UI
src/main.js               app wiring: ingest queue, search modes, viewer, exports
src/pdf-extract.js        pdf.js text layer -> runs -> lines; page rendering
src/ocr.js, ocr.worker.js OCR + reading-order pipeline (onnxruntime-web)
src/regex.js, search.js   regex -> NFA tables; JS matcher
src/gpu-search.js         WebGPU kernels (regex + fuzzy) and buffers
src/fuzzy.js              fuzzy matcher: normalization, JS reference, WGSL
src/highlight.js          match -> glyph-accurate rects on the source page
src/semantic.js           passage chunking, vector store, ranking
src/embed.worker.js       EmbeddingGemma download/cache + inference
src/store.js              line store (code-point cells), IndexedDB, OPFS, dumps
vendor/                   pdf.js, onnxruntime-web, hyparquet-writer, tokenizers.js
```

## Third-party

- [pdf.js](https://github.com/mozilla/pdf.js) (Apache-2.0)
- [onnxruntime-web](https://github.com/microsoft/onnxruntime) 1.30 (MIT). Only the `asyncify` build is used.
- [hyparquet-writer](https://github.com/hyparam/hyparquet-writer) (MIT), bundled as `vendor/hyparquet-writer.min.mjs`
- [tokenizers.js](https://github.com/huggingface/tokenizers.js) (Apache-2.0), bundled as `vendor/tokenizers.min.mjs`
- [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma) is fetched at runtime from [onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) and is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). It isn't redistributed in this repository.
