// Port of indra_ocr/order_post.py + layout_post.py (torch-free post-processing).
// All arrays are flat row-major Float32/Float64 arrays unless noted.
import { median } from "./image.js";

export const N_GEOM = 10, N_TEXT = 6, N_FEATS = 16;

export const BLOCK_CLASSES = [
  "Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group",
  "Table", "Caption", "Footnote", "Table-Of-Contents", "Form",
];
const CLASS_ID = Object.fromEntries(BLOCK_CLASSES.map((c, i) => [c, i]));
const GRID_LABELS = new Set(["Table", "Form", "Table-Of-Contents"]);
const GRID = new Set([...GRID_LABELS].map((c) => CLASS_ID[c]));

const HEADER = CLASS_ID["Page-Header"], FOOTER = CLASS_ID["Page-Footer"];
const SECTION = CLASS_ID["Section-Header"], LIST = CLASS_ID["List-Group"];
const TABLE = CLASS_ID["Table"];

// ---------------------------------------------------------------------------
// line features (order_post.line_features)
// ---------------------------------------------------------------------------
export function lineFeatures(boxes, W, H, texts) {
  const n = boxes.length / 4;
  const out = new Float32Array(n * N_FEATS);
  if (!n) return out;
  const hs = new Float64Array(n);
  for (let i = 0; i < n; i++) hs[i] = Math.max(boxes[4 * i + 3] - boxes[4 * i + 1], 1e-3);
  const medH = median(hs);
  for (let i = 0; i < n; i++) {
    const x0 = boxes[4 * i], y0 = boxes[4 * i + 1], x1 = boxes[4 * i + 2], y1 = boxes[4 * i + 3];
    const w = Math.max(x1 - x0, 1e-3), h = Math.max(y1 - y0, 1e-3);
    const o = i * N_FEATS;
    out[o] = x0 / W; out[o + 1] = y0 / H; out[o + 2] = x1 / W; out[o + 3] = y1 / H;
    out[o + 4] = (x0 + x1) / (2 * W); out[o + 5] = (y0 + y1) / (2 * H);
    out[o + 6] = w / W; out[o + 7] = Math.log(h / medH);
    out[o + 8] = Math.log(w / h) / 4; out[o + 9] = (medH / H) * 20;
    const t = (texts[i] || "").trim();
    if (t) {
      let digits = 0;
      for (const c of t) if (c >= "0" && c <= "9") digits++;
      const first3 = t.slice(0, 3).replace(/[.)]/g, "");
      out[o + 10] = Math.log1p(t.length) / 5;
      out[o + 11] = digits / t.length;
      out[o + 12] = ("•-–▪◦*·→".includes(t[0]) || (/^\d+$/.test(first3) && t.length > 3)) ? 1 : 0;
      out[o + 13] = t.endsWith("-") ? 1 : 0;
      out[o + 14] = ".!?:;".includes(t[t.length - 1]) ? 1 : 0;
      out[o + 15] = /^[A-Z]/.test(t[0]) ? 1 : 0;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// decode_order — row-sum ranking + insertion local search
// ---------------------------------------------------------------------------
// lp(i before j) = log sigmoid(a) = -softplus(-a)
const softplus = (x) => (x > 30 ? x : x < -30 ? 0 : Math.log1p(Math.exp(x)));

export function decodeOrder(a, n, sweeps = 4) {
  if (n <= 1) return [...Array(n).keys()];
  const lp = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) lp[i] = -softplus(-a[i]);
  const score = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) score[i] += -lp[i * n + j] + lp[j * n + i];
  let order = [...Array(n).keys()].sort((x, y) => score[x] - score[y]);
  for (let s = 0; s < sweeps; s++) {
    let improved = false;
    for (const x of [...order]) {
      const rest = order.filter((v) => v !== x);
      const m = rest.length;
      const gains = new Float64Array(m + 1);
      let acc = 0;
      for (let k = 0; k < m; k++) { acc += lp[rest[k] * n + x]; gains[k + 1] += acc; }
      acc = 0;
      for (let k = m - 1; k >= 0; k--) { acc += lp[x * n + rest[k]]; gains[k] += acc; }
      let k = 0;
      for (let i = 1; i <= m; i++) if (gains[i] > gains[k]) k = i;
      const cur = order.indexOf(x);
      if (gains[k] > gains[cur] + 1e-6) {
        rest.splice(k, 0, x); order = rest; improved = true;
      }
    }
    if (!improved) break;
  }
  return order;
}

// ---------------------------------------------------------------------------
// order_top_down / order_xycut baselines
// ---------------------------------------------------------------------------
export function orderTopDown(boxes) {
  const n = boxes.length / 4;
  if (!n) return [];
  const hs = [];
  for (let i = 0; i < n; i++) hs.push(boxes[4 * i + 3] - boxes[4 * i + 1]);
  const h = median(hs);
  const cy = (i) => (boxes[4 * i + 1] + boxes[4 * i + 3]) / 2;
  return [...Array(n).keys()].sort(
    (a, b) => Math.round(cy(a) / Math.max(h * 0.5, 1e-3)) - Math.round(cy(b) / Math.max(h * 0.5, 1e-3)) || boxes[4 * a] - boxes[4 * b]
  );
}

export function orderXycut(boxes) {
  const n = boxes.length / 4;
  if (!n) return [];
  const bestCut = (ids, axis) => {
    const lo = (i) => boxes[4 * i + (axis === 1 ? 1 : 0)];
    const hi = (i) => boxes[4 * i + (axis === 1 ? 3 : 2)];
    const o = [...ids].sort((a, b) => lo(a) - lo(b));
    let end = hi(o[0]), bestGap = 0, bestK = -1;
    for (let k = 1; k < o.length; k++) {
      const gap = lo(o[k]) - end;
      if (gap > bestGap) { bestGap = gap; bestK = k; }
      end = Math.max(end, hi(o[k]));
    }
    return { gap: bestGap, k: bestK, o };
  };
  const rec = (ids) => {
    if (ids.length <= 1) return ids;
    const sub = (sel) => {
      const b = new Float32Array(sel.length * 4);
      sel.forEach((v, k) => b.set(boxes.subarray(4 * v, 4 * v + 4), 4 * k));
      return b;
    };
    const gy = bestCut(ids, 1), gx = bestCut(ids, 0);
    if (gy.k < 0 && gx.k < 0) {
      const b = sub(ids);
      return orderTopDown(b).map((i) => ids[i]);
    }
    const { k, o } = gy.gap >= gx.gap ? gy : gx;
    return rec(o.slice(0, k)).concat(rec(o.slice(k)));
  };
  return rec([...Array(n).keys()]);
}

// ---------------------------------------------------------------------------
// _cluster — average-linkage clustering of "same group" logits [n*n]
// ---------------------------------------------------------------------------
export function cluster(logits, n) {
  if (!n) return [];
  const S = Float64Array.from(logits);
  for (let i = 0; i < n; i++) S[i * n + i] = 0;
  const size = new Float64Array(n).fill(1);
  const alive = new Uint8Array(n).fill(1);
  const members = [...Array(n).keys()].map((i) => [i]);
  const M = Float64Array.from(S);
  for (let i = 0; i < n; i++) M[i * n + i] = -Infinity;
  const best = new Int32Array(n), bestV = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let bi = 0;
    for (let j = 1; j < n; j++) if (M[i * n + j] > M[i * n + bi]) bi = j;
    best[i] = bi; bestV[i] = M[i * n + bi];
  }
  for (let iter = 0; iter < n - 1; iter++) {
    let a = 0;
    for (let i = 1; i < n; i++) if (bestV[i] > bestV[a]) a = i;
    if (!(bestV[a] > 0)) break;
    let b = best[a];
    if (a > b) [a, b] = [b, a];
    for (let j = 0; j < n; j++) {
      S[a * n + j] += S[b * n + j];
      S[j * n + a] += S[j * n + b];
    }
    S[a * n + a] = 0;
    size[a] += size[b]; alive[b] = 0;
    members[a] = members[a].concat(members[b]);
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) row[j] = alive[j] && j !== a ? S[a * n + j] / (size[a] * size[j]) : -Infinity;
    for (let j = 0; j < n; j++) { M[a * n + j] = row[j]; M[j * n + a] = row[j]; }
    for (let j = 0; j < n; j++) { M[b * n + j] = -Infinity; M[j * n + b] = -Infinity; }
    bestV[b] = -Infinity;
    let bi = 0;
    for (let j = 1; j < n; j++) if (row[j] > row[bi]) bi = j;
    best[a] = bi; bestV[a] = row[bi];
    for (let j = 0; j < n; j++) {
      if (!alive[j] || j === a) continue;
      if (best[j] === a || best[j] === b || row[j] > bestV[j]) {
        let bj = 0;
        for (let k = 1; k < n; k++) if (M[j * n + k] > M[j * n + bj]) bj = k;
        if (row[j] > bestV[j]) { best[j] = a; bestV[j] = row[j]; }
        else { best[j] = bj; bestV[j] = M[j * n + bj]; }
      }
    }
  }
  return members.filter((_, i) => alive[i]).map((m) => m.sort((x, y) => x - y));
}

// ---------------------------------------------------------------------------
// decode_blocks_order
// ---------------------------------------------------------------------------
export function decodeBlocksOrder(a, same, boxes, cls, n) {
  if (n <= 1) return [...Array(n).keys()];
  const blocks = cluster(same, n);
  const inner = blocks.map((b) => {
    const bx = new Float32Array(b.length * 4);
    b.forEach((v, k) => bx.set(boxes.subarray(4 * v, 4 * v + 4), 4 * k));
    let isGrid = false;
    if (cls) {
      // argmax of summed class logits over the block's lines
      const C = cls.length / n;
      const sum = new Float64Array(C);
      for (const v of b) for (let c = 0; c < C; c++) sum[c] += cls[v * C + c];
      let am = 0;
      for (let c = 1; c < C; c++) if (sum[c] > sum[am]) am = c;
      isGrid = GRID.has(am);
    }
    const local = isGrid ? orderTopDown(bx) : orderXycut(bx);
    return local.map((k) => b[k]);
  });
  const m = blocks.length;
  // mean pairwise order logit between blocks, antisymmetrised
  const ab = new Float32Array(m * m);
  for (let p = 0; p < m; p++)
    for (let q = p + 1; q < m; q++) {
      let s = 0;
      for (const i of blocks[p]) for (const j of blocks[q]) s += a[i * n + j];
      s /= blocks[p].length * blocks[q].length;
      ab[p * m + q] = s; ab[q * m + p] = -s;
    }
  const bo = decodeOrder(ab, m);
  return bo.flatMap((k) => inner[k]);
}

// ---------------------------------------------------------------------------
// textlayer.py bits: pdf order coherence (used by text path)
// ---------------------------------------------------------------------------
export function orderBacktracks(boxes) {
  const n = boxes.length / 4;
  if (n < 3) return 0;
  const hs = [];
  for (let i = 0; i < n; i++) hs.push(boxes[4 * i + 3] - boxes[4 * i + 1]);
  const h = median(hs);
  let bt = 0;
  for (let i = 0; i + 1 < n; i++) {
    const a = boxes.subarray(4 * i, 4 * i + 4), b = boxes.subarray(4 * i + 4, 4 * i + 8);
    if (b[1] < a[1] - 0.5 * h && b[0] < a[2] - h) bt++;
  }
  return bt;
}

export const MIN_MODEL_AGREEMENT = 0.85;

export function pdfOrderCoherent(boxes, pairLogits) {
  const n = boxes.length / 4;
  if (orderBacktracks(boxes) > Math.max(1, Math.floor(0.02 * n))) return false;
  if (!pairLogits || n < 3) return true;
  let agree = 0;
  for (let i = 0; i + 1 < n; i++) if (pairLogits[i * n + i + 1] > 0) agree++;
  return agree / (n - 1) >= MIN_MODEL_AGREEMENT;
}

// ---------------------------------------------------------------------------
// layout_post.py: markdown assembly
// ---------------------------------------------------------------------------
const WHOLE_MATH = /^\$[^$]+\$$/;
const ITEM = /^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)/;

function cjk(cp) {
  return (cp >= 0x3000 && cp <= 0x30ff) || (cp >= 0x3400 && cp <= 0x9fff) ||
    (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0xff00 && cp <= 0xffef) ||
    (cp >= 0x20000 && cp <= 0x2fa1f);
}
const lastCp = (s) => s.length ? s.codePointAt(s.length - 1) : 0;
const firstCp = (s) => s.length ? s.codePointAt(0) : 0;

export function joinLines(texts) {
  let out = "";
  for (let t of texts) {
    t = t.trim();
    if (!t) continue;
    const last = out.slice(-1), first = t[0];
    if (
      out.length > 1 && out.endsWith("-") && /[a-zA-Z]/.test(out[out.length - 2]) &&
      /[a-z]/.test(first)
    ) {
      out = out.slice(0, -1) + t;
    } else if (out && cjk(lastCp(out)) && cjk(firstCp(t))) {
      out += t;
    } else {
      out = out ? out + " " + t : t;
    }
  }
  return out;
}

export function splitBlocks(order, same, n) {
  const blocks = [];
  for (const i of order) {
    const lastB = blocks[blocks.length - 1];
    if (lastB && same[lastB[lastB.length - 1] * n + i] > 0) lastB.push(i);
    else blocks.push([i]);
  }
  return blocks;
}

function tableRows(boxes) {
  const n = boxes.length / 4;
  const cy = (i) => (boxes[4 * i + 1] + boxes[4 * i + 3]) / 2;
  const hs = [];
  for (let i = 0; i < n; i++) hs.push(boxes[4 * i + 3] - boxes[4 * i + 1]);
  const h = median(hs);
  const idx = [...Array(n).keys()].sort((a, b) => cy(a) - cy(b));
  const rows = [];
  for (const i of idx) {
    const last = rows[rows.length - 1];
    const meanCy = last ? last.reduce((s, k) => s + cy(k), 0) / last.length : 0;
    if (last && Math.abs(cy(i) - meanCy) < 0.5 * h) last.push(i);
    else rows.push([i]);
  }
  return rows.map((r) => r.sort((a, b) => boxes[4 * a] - boxes[4 * b]));
}

function mergeWrapped(rows, boxes) {
  if (rows.length < 3) return rows;
  const n = boxes.length / 4;
  const hs = [];
  for (let i = 0; i < n; i++) hs.push(boxes[4 * i + 3] - boxes[4 * i + 1]);
  const h = median(hs);
  const gaps = rows.slice(1).map((r, k) => {
    const cur = rows[k];
    const minY1 = Math.max(...cur.map((i) => boxes[4 * i + 3]));
    const minY0 = Math.min(...r.map((i) => boxes[4 * i + 1]));
    return minY0 - minY1;
  });
  const med = median(gaps);
  const out = [[...rows[0]]];
  let prev = rows[0];
  for (let k = 0; k < gaps.length; k++) {
    const g = gaps[k], r = rows[k + 1];
    if (g < 0.5 * med && g < 0.6 * h && r.length < prev.length) out[out.length - 1].push(...r);
    else out.push([...r]);
    prev = r;
  }
  return out;
}

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function tableHtml(boxes, texts) {
  let rows = tableRows(boxes);
  const x0min = Math.min(...[...Array(texts.length).keys()].map((i) => boxes[4 * i]));
  const x1max = Math.max(...[...Array(texts.length).keys()].map((i) => boxes[4 * i + 2]));
  const spanX = x1max - x0min;
  if (
    rows.length > 2 && rows[0].length === 1 &&
    Math.max(...rows.map((r) => r.length)) > 1 &&
    boxes[4 * rows[0][0] + 2] - boxes[4 * rows[0][0]] > 0.6 * spanX
  ) {
    const keep = [...Array(texts.length).keys()].filter((k) => k !== rows[0][0]);
    const kb = new Float32Array(keep.length * 4);
    keep.forEach((v, i) => kb.set(boxes.subarray(4 * v, 4 * v + 4), 4 * i));
    return texts[rows[0][0]] + "\n\n" + tableHtml(kb, keep.map((k) => texts[k]));
  }
  const width = Math.max(...rows.map((r) => r.length));
  const full = rows.filter((r) => r.length === width);
  rows = mergeWrapped(rows, boxes);
  const bands = [];
  for (let c = 0; c < width; c++) {
    const l = median(full.map((r) => boxes[4 * r[c]]));
    const r = median(full.map((r) => boxes[4 * r[c] + 2]));
    bands.push([l, r]);
  }
  const edges = [-Infinity, ...bands.slice(0, -1).map((b, c) => (b[1] + bands[c + 1][0]) / 2), Infinity];
  const bandOf = (x) => {
    for (let c = 0; c < width; c++) if (x < edges[c + 1]) return c;
    return width - 1;
  };
  const out = ["<table>"];
  rows.forEach((r, ri) => {
    const tag = ri === 0 && rows.length > 1 ? "th" : "td";
    const cells = new Map();
    for (const k of [...r].sort((a, b) => boxes[4 * a + 1] - boxes[4 * b + 1] || boxes[4 * a] - boxes[4 * b])) {
      const bx0 = boxes[4 * k], bx1 = boxes[4 * k + 2];
      const first = bandOf(bx0), last = bandOf(bx1);
      let span = [];
      for (let c = Math.max(first, 0); c <= Math.min(last, width - 1); c++) {
        if (Math.min(bx1, bands[c][1]) - Math.max(bx0, bands[c][0]) > 0.5 * (bands[c][1] - bands[c][0]))
          span.push(c);
      }
      if (!span.length) span = [Math.min(Math.max(first, 0), width - 1)];
      const c0 = span[0];
      const prev = cells.get(c0);
      const text = prev ? prev[1] + " " + texts[k] : texts[k];
      cells.set(c0, [prev ? Math.max(prev[0], span.length) : span.length, text]);
    }
    const tds = [];
    let c = 0;
    while (c < width) {
      if (cells.has(c)) {
        const [n2, text] = cells.get(c);
        tds.push(`<${tag}${n2 > 1 ? ` colspan="${n2}"` : ""}>${esc(text)}</${tag}>`);
        c += n2;
      } else { tds.push(`<${tag}></${tag}>`); c++; }
    }
    out.push("<tr>" + tds.join("") + "</tr>");
  });
  out.push("</table>");
  return out.join("\n");
}

function headsPlausible(boxes, tab, n) {
  const cells = cluster(tab[2], n);
  let bad = 0;
  for (const c of cells) {
    if (c.length < 2) continue;
    let ok = true;
    for (let a = 0; a < c.length && ok; a++)
      for (let b = a + 1; b < c.length; b++) {
        const x0a = boxes[4 * c[a]], x1a = boxes[4 * c[a] + 2];
        const x0b = boxes[4 * c[b]], x1b = boxes[4 * c[b] + 2];
        if (Math.min(x1a, x1b) - Math.max(x0a, x0b) < 0) { ok = false; break; }
      }
    if (!ok) bad++;
  }
  return bad <= 0.1 * cells.length;
}

function tableHtmlHeads(boxes, texts, tab, n) {
  // tab: [3][n][n] flat — same row / column / cell logits
  const cells = cluster(tab[2], n);
  const cb = cells.map((c) => [
    Math.min(...c.map((k) => boxes[4 * k])), Math.min(...c.map((k) => boxes[4 * k + 1])),
    Math.max(...c.map((k) => boxes[4 * k + 2])), Math.max(...c.map((k) => boxes[4 * k + 3])),
  ]);
  const ctext = cells.map((c) =>
    [...c].sort((a, b) => boxes[4 * a + 1] - boxes[4 * b + 1] || boxes[4 * a] - boxes[4 * b])
      .map((k) => texts[k]).join(" "));
  const m = cells.length;
  const cellLogits = (k) => {
    const out = new Float32Array(m * m);
    for (let a = 0; a < m; a++)
      for (let b = a + 1; b < m; b++) {
        let s = 0;
        for (const i of cells[a]) for (const j of cells[b]) s += tab[k][i * n + j];
        out[a * m + b] = out[b * m + a] = s / (cells[a].length * cells[b].length);
      }
    return out;
  };
  const rows = cluster(cellLogits(0), m).sort((a, b) => Math.min(...a.map((c) => cb[c][1])) - Math.min(...b.map((c) => cb[c][1])));
  const cols = cluster(cellLogits(1), m).sort((a, b) => Math.min(...a.map((c) => cb[c][0])) - Math.min(...b.map((c) => cb[c][0])));
  const rowOf = new Map(), colOf = new Map();
  rows.forEach((g, r) => g.forEach((c) => rowOf.set(c, r)));
  cols.forEach((g, k) => g.forEach((c) => colOf.set(c, k)));
  const centers = cols.map((g) => median(g.map((c) => (cb[c][0] + cb[c][2]) / 2)));
  const hs = [];
  for (let i = 0; i < n; i++) hs.push(boxes[4 * i + 3] - boxes[4 * i + 1]);
  const h = median(hs);
  const grid = new Map();
  for (let c = 0; c < m; c++) {
    const c0 = colOf.get(c);
    let span = 1;
    while (c0 + span < cols.length && cb[c][0] - h <= centers[c0 + span] && centers[c0 + span] <= cb[c][2] + h) span++;
    const key = rowOf.get(c) * 4096 + c0;
    if (grid.has(key)) {
      const g = grid.get(key);
      grid.set(key, [Math.max(g[0], span), g[1] + " " + ctext[c]]);
    } else grid.set(key, [span, ctext[c]]);
  }
  const out = ["<table>"];
  for (let r = 0; r < rows.length; r++) {
    const tag = r === 0 && rows.length > 1 ? "th" : "td";
    const tds = [];
    let c = 0;
    while (c < cols.length) {
      const key = r * 4096 + c;
      if (grid.has(key)) {
        const [span, text] = grid.get(key);
        tds.push(`<${tag}${span > 1 ? ` colspan="${span}"` : ""}>${esc(text)}</${tag}>`);
        c += span;
      } else { tds.push(`<${tag}></${tag}>`); c++; }
    }
    out.push("<tr>" + tds.join("") + "</tr>");
  }
  out.push("</table>");
  return out.join("\n");
}

export function pageBlocks(boxes, texts, order, clsLogits, same, n) {
  const C = clsLogits.length / n;
  return splitBlocks(order, same, n).map((blk) => {
    const sum = new Float64Array(C);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const i of blk) {
      for (let c = 0; c < C; c++) sum[c] += clsLogits[i * C + c];
      x0 = Math.min(x0, boxes[4 * i]); y0 = Math.min(y0, boxes[4 * i + 1]);
      x1 = Math.max(x1, boxes[4 * i + 2]); y1 = Math.max(y1, boxes[4 * i + 3]);
    }
    let am = 0;
    for (let c = 1; c < C; c++) if (sum[c] > sum[am]) am = c;
    return {
      label: BLOCK_CLASSES[am],
      bbox: [x0, y0, x1, y1].map((v) => Math.round(v * 10) / 10),
      text: blk.map((i) => texts[i]).join("\n"),
    };
  });
}

/** clsLogits [n*C], same [n*n], cells: Map line idx -> [[box,text]...], tab: [3,n,n]|null */
export function pageMarkdown(boxes, texts, order, clsLogits, same, n, opts = {}) {
  const { keepFurniture = false, cells = null, tab = null } = opts;
  const C = clsLogits.length / n;
  const parts = [];
  for (const blk of splitBlocks(order, same, n)) {
    const sum = new Float64Array(C);
    for (const i of blk) for (let c = 0; c < C; c++) sum[c] += clsLogits[i * C + c];
    let cls = 0;
    for (let c = 1; c < C; c++) if (sum[c] > sum[cls]) cls = c;
    const tb = blk.map((i) => texts[i]);
    if ((cls === HEADER || cls === FOOTER) && !keepFurniture) continue;
    if (tb.every((t) => WHOLE_MATH.test(t.trim()))) {
      parts.push(tb.map((t) => "$$" + t.trim().slice(1, -1) + "$$").join("\n"));
      continue;
    }
    if (cls === SECTION) {
      parts.push("## " + joinLines(tb));
    } else if (cls === TABLE && blk.length > 1) {
      const sub = new Float32Array(blk.length * 4);
      blk.forEach((v, k) => sub.set(boxes.subarray(4 * v, 4 * v + 4), 4 * k));
      const subTab = tab
        ? [0, 1, 2].map((k) => {
            const m = new Float32Array(blk.length * blk.length);
            blk.forEach((i2, a) => blk.forEach((j2, b) => (m[a * blk.length + b] = tab[k][i2 * n + j2])));
            return m;
          })
        : null;
      if (subTab && headsPlausible(sub, subTab, blk.length)) {
        parts.push(tableHtmlHeads(sub, tb, subTab, blk.length));
      } else {
        const cb = [], ct = [];
        for (const i of blk) {
          const cs = cells && cells.get(i);
          if (cs) for (const [bx, tx] of cs) { cb.push(...bx); ct.push(tx); }
          else { cb.push(...boxes.subarray(4 * i, 4 * i + 4)); ct.push(texts[i]); }
        }
        parts.push(tableHtml(Float32Array.from(cb), ct));
      }
    } else if (cls === LIST) {
      const items = [];
      for (const t of tb) {
        if (!items.length || ITEM.test(t)) items.push([t]);
        else items[items.length - 1].push(t);
      }
      parts.push(items.map((it) => joinLines(it)).join("\n"));
    } else {
      parts.push(joinLines(tb));
    }
  }
  return parts.filter(Boolean).join("\n\n");
}
