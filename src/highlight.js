// Precise match highlights on the source page.
//
// A stored line is runsToLines' join of text-layer runs, so for text-layer
// pages we re-read the page's runs, rebuild the same line string while keeping
// a char -> (run, offset) map, and measure glyph advances in the run's font to
// get sub-run x extents. OCR lines (or anything that doesn't rebuild exactly)
// fall back to proportional placement inside the line box.
import { pageTextRuns } from "./pdf-extract.js";

const measureCtx = new OffscreenCanvas(1, 1).getContext("2d");
function advance(font, s) {
  measureCtx.font = `100px ${font}`;
  return measureCtx.measureText(s).width;
}

/** Runs of the line whose box is `box` (they lie inside it), in x order. */
function lineRuns(runs, box) {
  const eps = 0.75;
  return runs
    .filter((r) => r.box[0] >= box[0] - eps && r.box[2] <= box[2] + eps &&
      r.box[1] >= box[1] - eps && r.box[3] <= box[3] + eps)
    .sort((a, b) => a.box[0] - b.box[0]);
}

/** Rebuild runsToLines' text: join(" "), collapse whitespace, trim. */
function rebuild(runs) {
  let text = "";
  const map = [];        // string index -> [runIdx, charIdx] | null
  let pendingSpace = false;
  runs.forEach((r, ri) => {
    if (ri > 0) pendingSpace = true;
    for (let ci = 0; ci < r.text.length; ci++) {
      const ch = r.text[ci];
      if (/\s/.test(ch)) { pendingSpace = true; continue; }
      if (pendingSpace && text.length) { text += " "; map.push(null); }
      pendingSpace = false;
      text += ch; map.push([ri, ci]);
    }
  });
  return { text, map };
}

// Text-layer boxes span baseline -> baseline + declared font size, which on
// OCR'd scans is often taller than the printed glyphs. Clamp highlight height
// to the page's baseline pitch and hang it from the baseline.
const pitchCache = new WeakMap();
function pagePitch(runs) {
  if (pitchCache.has(runs)) return pitchCache.get(runs);
  const base = runs.filter((r) => r.horiz).map((r) => r.box[3]).sort((a, b) => a - b);
  const gaps = [];
  for (let i = 1; i < base.length; i++) {
    const g = base[i] - base[i - 1];
    if (g > 2) gaps.push(g);
  }
  gaps.sort((a, b) => a - b);
  const p = gaps.length ? gaps[gaps.length >> 1] : Infinity;
  pitchCache.set(runs, p);
  return p;
}

function vertical(r, runs) {
  if (!r.horiz) return [r.box[1], r.box[3]];
  const h = Math.min(r.box[3] - r.box[1], pagePitch(runs));
  return [r.box[3] - 0.8 * h, r.box[3] + 0.2 * h];
}

function runRect(r, i0, i1, runs) {
  const [y0, y1] = vertical(r, runs);
  if (!r.horiz || r.text.length < 2) return [r.box[0], y0, r.box[2], y1];
  const full = advance(r.font, r.text) || 1;
  const k = (r.box[2] - r.box[0]) / full;
  const x0 = r.box[0] + advance(r.font, r.text.slice(0, i0)) * k;
  const x1 = r.box[0] + advance(r.font, r.text.slice(0, i1)) * k;
  return [x0, y0, x1, y1];
}

/** Proportional fallback inside a line box. */
function boxRect(text, box, a, b) {
  const font = "sans-serif";
  const full = advance(font, text) || 1;
  const k = (box[2] - box[0]) / full;
  return [box[0] + advance(font, text.slice(0, a)) * k, box[1], box[0] + advance(font, text.slice(0, b)) * k, box[3]];
}

/**
 * Page runs at `viewport` (same 200dpi space as stored boxes), or null for
 * pages without a usable text layer.
 */
export async function pageRuns(page, viewport) {
  try {
    const { runs } = await pageTextRuns(page, viewport);
    return runs.length ? runs : null;
  } catch {
    return null;
  }
}

/**
 * Highlight rects for string ranges [[a,b), ...] of one stored line.
 * Returns [{rect:[x0,y0,x1,y1], exact:boolean}].
 */
export function lineRangeRects(runs, line, ranges) {
  const out = [];
  if (!line.box) return out;
  let lr = null, rb = null;
  if (runs) {
    lr = lineRuns(runs, line.box);
    rb = lr.length ? rebuild(lr) : null;
  }
  const exact = rb && rb.text === line.text;
  for (const [a, b] of ranges) {
    if (b <= a) continue;
    if (!exact) { out.push({ rect: boxRect(line.text, line.box, a, b), exact: false }); continue; }
    // walk chars, emitting one rect per contiguous run segment
    let cur = null;
    const flush = () => { if (cur) out.push({ rect: runRect(lr[cur.ri], cur.i0, cur.i1, runs), exact: true }); cur = null; };
    for (let i = a; i < b; i++) {
      const m = rb.map[i];
      if (!m) continue;
      const [ri, ci] = m;
      if (cur && cur.ri === ri && ci === cur.i1) cur.i1 = ci + 1;
      else { flush(); cur = { ri, i0: ci, i1: ci + 1 }; }
    }
    flush();
  }
  // merge rects that touch on the same line (word gaps between runs)
  out.sort((p, q) => p.rect[0] - q.rect[0]);
  const merged = [];
  for (const r of out) {
    const last = merged[merged.length - 1];
    const h = r.rect[3] - r.rect[1];
    if (last && r.rect[0] - last.rect[2] < 0.6 * h && Math.abs(r.rect[1] - last.rect[1]) < 0.3 * h) {
      last.rect[2] = Math.max(last.rect[2], r.rect[2]);
      last.rect[1] = Math.min(last.rect[1], r.rect[1]);
      last.rect[3] = Math.max(last.rect[3], r.rect[3]);
    } else merged.push({ rect: r.rect.slice(), exact: r.exact });
  }
  return merged;
}
