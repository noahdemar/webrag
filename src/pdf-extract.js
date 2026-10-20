// Main-thread pdf.js helpers: text-layer extraction -> line boxes (auto mode
// fast path) and page rendering -> grayscale for the OCR worker.
import * as pdfjs from "../vendor/pdfjs/pdf.min.mjs";
import { median } from "./image.js";

pdfjs.GlobalWorkerOptions.workerSrc =
  new URL("../vendor/pdfjs/pdf.worker.min.mjs", import.meta.url).href;

export { pdfjs };

const BAD_CHAR_MAX = 0.005;   // textlayer.py: MAX_BAD_CHAR_FRAC
const MIN_CHARS = 8;          // a layer with almost nothing is not a text layer

function badChars(s) {
  let n = 0;
  for (const c of s) {
    const cp = c.codePointAt(0);
    if (
      c === "�" ||
      (cp >= 0xe000 && cp <= 0xf8ff) ||        // private use
      (cp >= 0xf0000 && cp <= 0xffffd) ||
      (cp >= 0x100000 && cp <= 0x10fffd) ||
      (cp < 0x20 && c !== "\t" && c !== "\n") ||
      (cp >= 0x7f && cp <= 0x9f) ||
      (cp >= 0xfdd0 && cp <= 0xfdef) || (cp & 0xfffe) === 0xfffe
    ) n++;
  }
  return n;
}

const VENDOR = new URL("../vendor/pdfjs/", import.meta.url).href;

/** Open a PDF from an ArrayBuffer. */
export async function openPdf(data, opts = {}) {
  const task = pdfjs.getDocument({
    data,
    isEvalSupported: false,
    wasmUrl: VENDOR + "wasm/",
    cMapUrl: VENDOR + "cmaps/",
    standardFontDataUrl: VENDOR + "standard_fonts/",
    iccUrl: VENDOR + "iccs/",
    cMapPacked: true,
    ...opts,
  });
  const doc = await task.promise;
  doc.destroy = () => task.destroy();   // proxy has no destroy(); task does
  return doc;
}

/**
 * Text-layer items of one page as device-space boxes:
 * [{box:[x0,y0,x1,y1] px, text, contentIdx}] in PDF content order.
 * `viewport` must be the render viewport used elsewhere (scale k px/pt).
 */
export async function pageTextRuns(page, viewport) {
  const tc = await page.getTextContent({ includeMarkedContent: false });
  const vt = viewport.transform;
  const runs = [];
  let nChars = 0, nBad = 0;
  for (let i = 0; i < tc.items.length; i++) {
    const it = tc.items[i];
    const str = it.str;
    if (!str || !str.trim()) continue;
    nChars += str.length;
    nBad += badChars(str);
    // it.transform = text-space -> user-space matrix [a,b,c,d,e,f].
    // it.width/it.height are already USER-space units — extend them along the
    // transform's normalized axes (handles rotation), then map user -> device px.
    const [a, b, c, d, e, f] = it.transform;
    const la = Math.hypot(a, b) || 1, lc = Math.hypot(c, d) || 1;
    const w = it.width, h = it.height;
    const ux = a / la, uy = b / la, vx = c / lc, vy = d / lc;
    const pts = [
      [e, f],
      [e + w * ux, f + w * uy],
      [e + h * vx, f + h * vy],
      [e + w * ux + h * vx, f + w * uy + h * vy],
    ].map(([x, y]) => [
      vt[0] * x + vt[2] * y + vt[4],
      vt[1] * x + vt[3] * y + vt[5],
    ]);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    runs.push({
      box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
      text: str,
      contentIdx: i,
      hasEOL: !!it.hasEOL,
      font: tc.styles[it.fontName]?.fontFamily || "sans-serif",
      // baseline runs left-to-right in device space (char offsets map to x)
      horiz: pts[1][0] - pts[0][0] > 0 && Math.abs(pts[1][1] - pts[0][1]) < 0.02 * (pts[1][0] - pts[0][0]),
    });
  }
  return { runs, stats: { chars: nChars, bad: nBad } };
}

/** Is the text layer trustworthy? (bad char fraction, minimum size) */
export function textLayerOk({ runs, stats }) {
  if (!runs.length || stats.chars < MIN_CHARS) return false;
  return stats.bad <= Math.max(1, BAD_CHAR_MAX * stats.chars);
}

/**
 * Group text-layer runs into line boxes — approximates indra's assign_runs:
 * cluster by vertical centre (half a median height), split at wide x gaps.
 * Returns {boxes: Float32Array(4n), texts: string[], contentIdx: number[]}
 */
export function runsToLines(runs) {
  if (!runs.length) return { boxes: new Float32Array(0), texts: [], contentIdx: [] };
  const hs = runs.map((r) => r.box[3] - r.box[1]).sort((a, b) => a - b);
  const medH = median(hs.filter((v) => v > 0)) || 10;
  const cy = (r) => (r.box[1] + r.box[3]) / 2;
  const idx = [...runs.keys()].sort((a, b) => cy(runs[a]) - cy(runs[b]));
  const rows = [];
  for (const i of idx) {
    const r = runs[i];
    const last = rows[rows.length - 1];
    const meanCy = last ? last.reduce((s, k) => s + cy(runs[k]), 0) / last.length : 0;
    if (last && Math.abs(cy(r) - meanCy) < 0.5 * medH) last.push(i);
    else rows.push([i]);
  }
  const GAP = 2.2; // split a row at gaps wider than this many line heights
  const boxes = [], texts = [], contentIdx = [];
  for (const row of rows) {
    row.sort((a, b) => runs[a].box[0] - runs[b].box[0]);
    let cur = [];
    let curRight = -Infinity;
    for (const i of row) {
      const r = runs[i];
      if (cur.length && r.box[0] - curRight > GAP * medH) {
        flush();
      }
      cur.push(i);
      curRight = Math.max(curRight, r.box[2]);
    }
    flush();
    function flush() {
      if (!cur.length) return;
      const x0 = Math.min(...cur.map((i) => runs[i].box[0]));
      const y0 = Math.min(...cur.map((i) => runs[i].box[1]));
      const x1 = Math.max(...cur.map((i) => runs[i].box[2]));
      const y1 = Math.max(...cur.map((i) => runs[i].box[3]));
      const text = cur.map((i) => runs[i].text).join(" ").replace(/\s+/g, " ").trim();
      if (!text) { cur = []; curRight = -Infinity; return; }
      boxes.push(x0, y0, x1, y1);
      texts.push(text);
      contentIdx.push(Math.min(...cur.map((i) => runs[i].contentIdx)));
      cur = []; curRight = -Infinity;
    }
  }
  return { boxes: Float32Array.from(boxes), texts, contentIdx };
}

/** Render a page to grayscale Uint8Array at `scale` px/pt (canvas on main thread). */
export async function renderGray(page, scale, canvas, ctx) {
  const viewport = page.getViewport({ scale });
  const w = Math.ceil(viewport.width), h = Math.ceil(viewport.height);
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  await page.render({ canvasContext: ctx, viewport, background: "#ffffff" }).promise;
  const img = ctx.getImageData(0, 0, w, h);
  const rgba = img.data;
  const n = w * h;
  const gray = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    gray[i] = (77 * rgba[j] + 150 * rgba[j + 1] + 29 * rgba[j + 2]) >> 8;
  }
  return { gray, w, h };
}
