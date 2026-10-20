// Port of indra_ocr/pipeline.py + det_post.py + rec_preproc.py (worker side).
// Input is a grayscale Uint8Array page; ONNX sessions come from onnxruntime-web.
import {
  resizeGray, resizeBilinear, resizeArea, estimateSkew, rot90cw,
  unrotateBox, deskew, median,
} from "./image.js";
import {
  lineFeatures, decodeOrder, decodeBlocksOrder, orderXycut, orderTopDown,
  pageBlocks, pageMarkdown, pdfOrderCoherent, BLOCK_CLASSES, N_FEATS,
} from "./order_layout.js";

const ADAPTIVE_MIN_H = 9.0, ADAPTIVE_TARGET_H = 12.0, ADAPTIVE_MAX_SIDE = 2048;
const ORIENT_SIDE = 768, ORIENT_MIN = 0.95;
const SPLIT_GAP = 3.0, CELL_GAP = 1.8, GUTTER_GAP = 1.6, GUTTER_EDGES = 3;
const MIN_CONF = 0.5;
const LINE_HEIGHT = 32, PAD_Y = 0.2, PAD_X = 0.3, WIDTH_MULTIPLE = 8;
const INPUT_MULTIPLE = 32, OUT_STRIDE = 2;
const CORE_FRAC_X = 0.15;

// ---------------------------------------------------------------------------
// det.onnx — page_to_input + decode (det_post.py)
// ---------------------------------------------------------------------------

/** gray -> {x: Float32Array [ph*pw] ink map, scale, nh, nw} (pad to mult of 32). */
export function pageToInput(gray, w, h, longSide) {
  const scale = longSide / Math.max(h, w);
  let nh = Math.max(1, Math.round(h * scale));
  let nw = Math.max(1, Math.round(w * scale));
  const small = (nh !== h || nw !== w) ? resizeGray(gray, w, h, nw, nh) : gray;
  const ph = Math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const pw = Math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const x = new Float32Array(ph * pw);
  for (let y = 0; y < nh; y++) {
    const srow = y * nw, drow = y * pw;
    for (let xx = 0; xx < nw; xx++) x[drow + xx] = 1 - small[srow + xx] / 255;
  }
  return { x, scale, ph, pw };
}

/** Two-pass 4-connected components on a uint8 mask -> {lab, stats}.
 *  stats per label: {area, left, width}. lab: Int32Array [h*w]. */
function connectedComponents(mask, w, h) {
  const lab = new Int32Array(w * h);
  const parent = [0];
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
  let next = 1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      if (!mask[i]) continue;
      const up = y > 0 && mask[i - w] ? lab[i - w] : 0;
      const left = x > 0 && mask[i - 1] ? lab[i - 1] : 0;
      if (!up && !left) { lab[i] = next; parent[next] = next; next++; }
      else if (up && !left) lab[i] = up;
      else if (!up && left) lab[i] = left;
      else { lab[i] = Math.min(up, left); union(up, left); }
    }
  }
  // relabel to dense ids + stats
  const remap = new Int32Array(next);
  const stats = [];
  for (let i = 0; i < w * h; i++) {
    if (!lab[i]) continue;
    const r = find(lab[i]);
    if (!remap[r]) { remap[r] = stats.length + 1; stats.push({ area: 0, left: Infinity, width: 0 }); }
    lab[i] = remap[r];
  }
  // second pass for left/width: track min/max x per label
  const maxX = new Int32Array(stats.length + 1);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const l = lab[row + x];
      if (!l) continue;
      const st = stats[l - 1];
      st.area++;
      if (x < st.left) st.left = x;
      if (x + 1 > maxX[l]) maxX[l] = x + 1;
    }
  }
  stats.forEach((st, i) => { st.width = maxX[i + 1] - st.left; });
  return { lab, stats };
}

/** decode(detOutput, h2, w2) -> boxes [N*5] in input pixels (x0,y0,x1,y1,score). */
export function decode(out, w2, h2, thresh = 0.5, minArea = 2) {
  const npx = w2 * h2;
  const prob = new Float32Array(npx);
  const mask = new Uint8Array(npx);
  for (let i = 0; i < npx; i++) {
    const p = 1 / (1 + Math.exp(-out[i]));
    prob[i] = p;
    if (p > thresh) mask[i] = 1;
  }
  const { lab, stats } = connectedComponents(mask, w2, h2);
  const n = stats.length;
  if (!n) return new Float32Array(0);
  const topSum = new Float64Array(n), botSum = new Float64Array(n), scSum = new Float64Array(n);
  const cnt = new Float64Array(n);
  const topD = out.subarray(npx, 2 * npx), botD = out.subarray(2 * npx, 3 * npx);
  for (let y = 0; y < h2; y++) {
    const row = y * w2, yc = (y + 0.5) * OUT_STRIDE;
    for (let x = 0; x < w2; x++) {
      const i = row + x, l = lab[i];
      if (!l) continue;
      const k = l - 1;
      topSum[k] += yc - topD[i];
      botSum[k] += yc + botD[i];
      scSum[k] += prob[i];
      cnt[k]++;
    }
  }
  const boxes = [];
  for (let i = 0; i < n; i++) {
    const st = stats[i];
    if (st.area < minArea) continue;
    const y0 = topSum[i] / cnt[i], y1 = botSum[i] / cnt[i];
    const lh = Math.max(1, y1 - y0);
    const ex = Math.min(CORE_FRAC_X * lh, 0.5 * lh);
    boxes.push(
      st.left * OUT_STRIDE - ex, y0,
      (st.left + st.width) * OUT_STRIDE + ex, y1,
      scSum[i] / cnt[i],
    );
  }
  return Float32Array.from(boxes);
}

// ---------------------------------------------------------------------------
// rec: crop + batch + CTC decode (rec_preproc.py)
// ---------------------------------------------------------------------------

/** crop_line: expand box, crop w/ white fill, resize to LINE_HEIGHT. */
export function cropLine(gray, W, H, box, height = LINE_HEIGHT) {
  const [x0, y0, x1, y1] = box;
  const h = Math.max(1, y1 - y0);
  const X0 = x0 - PAD_X * h, X1 = x1 + PAD_X * h;
  const Y0 = y0 - PAD_Y * h, Y1 = y1 + PAD_Y * h;
  const ix0 = Math.floor(X0), iy0 = Math.floor(Y0);
  const ix1 = Math.ceil(X1), iy1 = Math.ceil(Y1);
  const cw = Math.max(1, ix1 - ix0), ch = Math.max(1, iy1 - iy0);
  const crop = new Uint8Array(cw * ch).fill(255);
  const sx0 = Math.max(0, ix0), sy0 = Math.max(0, iy0);
  const sx1 = Math.min(W, ix1), sy1 = Math.min(H, iy1);
  for (let y = sy0; y < sy1; y++) {
    const srow = y * W + sx0, drow = (y - iy0) * cw + (sx0 - ix0);
    for (let x = sx0; x < sx1; x++) crop[drow + x - sx0] = gray[srow + x - sx0];
  }
  const outW = Math.max(WIDTH_MULTIPLE, Math.min(4096, Math.round((cw * height) / ch)));
  const resized = ch > height
    ? resizeArea(crop, cw, ch, outW, height)
    : resizeBilinear(crop, cw, ch, outW, height);
  const ink = new Float32Array(outW * height);
  for (let i = 0; i < ink.length; i++) ink[i] = 1 - resized[i] / 255;
  return { ink, w: outW, ix0 };
}

export function padBatch(crops) {
  const H = LINE_HEIGHT;
  const Wm = Math.ceil(Math.max(...crops.map((c) => c.w)) / WIDTH_MULTIPLE) * WIDTH_MULTIPLE;
  const arr = new Float32Array(crops.length * H * Wm);
  const widths = new BigInt64Array(crops.length);
  crops.forEach((c, i) => {
    for (let y = 0; y < H; y++) {
      arr.set(c.ink.subarray(y * c.w, y * c.w + c.w), (i * H + y) * Wm);
    }
    widths[i] = BigInt(c.w);
  });
  return { arr, widths, Wm };
}

/** width_batches: group indices into batches of similar width. */
export function widthBatches(widths, maxBatch = 64, maxRatio = 1.3, minW = 64, maxArea = 64 * 640) {
  const order = [...widths.keys()].sort((a, b) => widths[a] - widths[b]);
  const batches = [];
  for (const i of order) {
    const last = batches[batches.length - 1];
    if (
      last && last.length < maxBatch &&
      Math.max(widths[i], minW) <= maxRatio * Math.max(widths[last[0]], minW) &&
      (last.length + 1) * Math.max(widths[i], minW) <= maxArea
    ) last.push(i);
    else batches.push([i]);
  }
  return batches;
}

// ---------------------------------------------------------------------------
// orient.onnx input
// ---------------------------------------------------------------------------
export function orientInput(gray, w, h, side = ORIENT_SIDE) {
  const s = side / Math.max(h, w);
  const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
  const small = resizeArea(gray, w, h, nw, nh);
  const x = new Float32Array(side * side);
  for (let y = 0; y < nh; y++) {
    const srow = y * nw, drow = y * side;
    for (let i = 0; i < nw; i++) x[drow + i] = 1 - small[srow + i] / 255;
  }
  return x;
}

// ---------------------------------------------------------------------------
// The pipeline: orient -> deskew -> det -> rec -> split -> order -> markdown
// ---------------------------------------------------------------------------
export class OcrEngine {
  /** sessions: {det, rec, order?, orient?, script?, recCjk?}; meta: meta.json */
  constructor(ort, sessions, meta) {
    this.ort = ort;
    this.s = sessions;
    // meta.vocab = JSON string {"blank":0,"chars":[...]}; CTC id i (>=1) -> chars[i-1]
    const un = (v) => {
      if (Array.isArray(v)) return v;
      if (typeof v === "string") v = JSON.parse(v);
      return v.chars;
    };
    this.vocab = un(meta.vocab);
    this.vocabCjk = meta.vocab_cjk ? un(meta.vocab_cjk) : null;
    this.longSide = meta.det_long_side;
    this.height = meta.rec_height;
    this.spaceId = this.vocab.indexOf(" ") + 1 || -1;
  }

  async run(sess, feeds) {
    const outs = await sess.run(feeds);
    return sess.outputNames.map((n) => outs[n]);
  }

  async detectAt(gray, w, h, longSide) {
    const { x, scale, ph, pw } = pageToInput(gray, w, h, longSide);
    const t = new this.ort.Tensor("float32", x, [1, 1, ph, pw]);
    const [out] = await this.run(this.s.det, { x: t });
    return { boxes: decode(out.data, pw / 2, ph / 2), scale };
  }

  async detect(gray, w, h) {
    let { boxes, scale } = await this.detectAt(gray, w, h, this.longSide);
    const n = boxes.length / 5;
    if (n >= 8) {
      const hs = [];
      for (let i = 0; i < n; i++) hs.push(boxes[5 * i + 3] - boxes[5 * i + 1]);
      const medH = median(hs);
      if (medH < ADAPTIVE_MIN_H) {
        const ls = Math.floor(Math.min(ADAPTIVE_MAX_SIDE, (this.longSide * ADAPTIVE_TARGET_H) / Math.max(medH, 1)) / 32) * 32;
        if (ls > this.longSide) ({ boxes, scale } = await this.detectAt(gray, w, h, ls));
      }
    }
    const n2 = boxes.length / 5;
    for (let i = 0; i < n2; i++) {
      boxes[5 * i] = Math.min(Math.max(boxes[5 * i] / scale, 0), w);
      boxes[5 * i + 1] = Math.min(Math.max(boxes[5 * i + 1] / scale, 0), h);
      boxes[5 * i + 2] = Math.min(Math.max(boxes[5 * i + 2] / scale, 0), w);
      boxes[5 * i + 3] = Math.min(Math.max(boxes[5 * i + 3] / scale, 0), h);
    }
    return boxes;
  }

  /** returns [{text, conf, toks:[[id,x],...], crop?}] per box. */
  async recognize(gray, W, H, boxes, batch = 64) {
    const n = boxes.length / 5;
    const crops = [];
    for (let i = 0; i < n; i++) crops.push(cropLine(gray, W, H, boxes.subarray(5 * i, 5 * i + 4), this.height));
    const out = new Array(n).fill(null).map(() => ({ text: "", conf: 0, toks: [] }));
    if (!n) return out;

    // script routing: which crops are hanzi -> rec_cjk
    let groups;
    if (this.s.script && this.s.recCjk && this.vocabCjk) {
      const hanzi = await this.hanziLines(crops);
      groups = [
        { sess: this.s.rec, vocab: this.vocab, sel: hanzi.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0), asChars: false },
        { sess: this.s.recCjk, vocab: this.vocabCjk, sel: hanzi.map((v, i) => (v ? i : -1)).filter((i) => i >= 0), asChars: true },
      ];
    } else {
      groups = [{ sess: this.s.rec, vocab: this.vocab, sel: [...Array(n).keys()], asChars: false }];
    }

    for (const { sess, vocab, sel, asChars } of groups) {
      for (const part of widthBatches(sel.map((j) => crops[j].w), batch)) {
        const idx = part.map((i) => sel[i]);
        const { arr, widths, Wm } = padBatch(idx.map((j) => crops[j]));
        const feeds = {
          x: new this.ort.Tensor("float32", arr, [idx.length, 1, this.height, Wm]),
          widths: new this.ort.Tensor("int64", widths, [idx.length]),
        };
        const outs = await sess.run(feeds);
        // outputs: (ids [B,T] int, pmax [B,T] f32, lengths [B] int) — identify by dtype/dims
        const ot = sess.outputNames.map((nm) => outs[nm]);
        const best = ot.find((t) => t.dims.length === 2 && t.type !== "float32");
        const pmax = ot.find((t) => t.dims.length === 2 && t.type === "float32");
        const lens = ot.find((t) => t.dims.length === 1);
        const T = best.dims[1];
        const stride = Wm / Math.max(T, 1);
        for (let k = 0; k < idx.length; k++) {
          const j = idx[k];
          const len = Number(lens.data[k]);
          const seq = Array.from(best.data.slice(k * T, k * T + len), Number);
          const pm = pmax.data.slice(k * T, k * T + len);
          const keep = [];
          for (let t2 = 0; t2 < seq.length; t2++)
            if (seq[t2] !== 0 && (t2 === 0 || seq[t2] !== seq[t2 - 1])) keep.push(t2);
          const [x0, y0, , y1] = [boxes[5 * j], boxes[5 * j + 1], 0, boxes[5 * j + 3]];
          const hh = Math.max(1, y1 - y0);
          const x1 = boxes[5 * j + 2];
          const ix0 = Math.floor(x0 - PAD_X * hh);
          const cw = Math.ceil(x1 + PAD_X * hh) - ix0;
          const f = cw / crops[j].w;
          const xs = keep.map((t2) => ix0 + (t2 + 0.5) * stride * f);
          const ids = keep.map((t2) => seq[t2]);
          const toks = asChars
            ? ids.map((id, i2) => [vocab[id - 1], xs[i2]])
            : ids.map((id, i2) => [id, xs[i2]]);
          const conf = keep.length ? keep.reduce((s, t2) => s + pm[t2], 0) / keep.length : 0;
          const text = conf >= MIN_CONF
            ? this.decodeIds(ids, vocab)
            : "";
          out[j] = { text, conf, toks: conf >= MIN_CONF ? toks : [] };
        }
      }
    }
    return out;
  }

  decodeIds(ids, vocab) {
    const toks = ids.filter((i) => i > 0 && i <= vocab.length).map((i) => vocab[i - 1]);
    const hasCmd = toks.some((t) => t.length > 1);
    if (!hasCmd) return toks.join("");
    let out = "";
    toks.forEach((t, k) => {
      out += t;
      const nxt = toks[k + 1] || "";
      if (t.length > 2 && t[0] === "\\" && /^[a-zA-Z]+$/.test(t.slice(1)) && /^[a-zA-Z]/.test(nxt)) out += " ";
    });
    return out;
  }

  async hanziLines(crops, batch = 64) {
    const out = new Array(crops.length).fill(false);
    for (const part of widthBatches(crops.map((c) => c.w), batch)) {
      const { arr, widths, Wm } = padBatch(part.map((i) => crops[i]));
      const res = await this.s.script.run({
        x: new this.ort.Tensor("float32", arr, [part.length, 1, this.height, Wm]),
        widths: new this.ort.Tensor("int64", widths, [part.length]),
      });
      const p = res[this.s.script.outputNames[0]].data;
      part.forEach((j, k) => (out[j] = p[k] > 0.5));
    }
    return out;
  }

  isSpace(tok) { return tok === " " || tok === this.spaceId; }

  /** x positions of gap cuts on column gutters (pipeline.gutter_cuts). */
  gutterCuts(boxes, rec) {
    const n = boxes.length / 4;
    const lefts = (i) => boxes[4 * i], rights = (i) => boxes[4 * i + 2];
    const cuts = [];
    for (let i = 0; i < n; i++) {
      const hh = Math.max(1, boxes[4 * i + 3] - boxes[4 * i + 1]);
      const xs = rec[i].toks.filter((tk) => !this.isSpace(tk[0])).map((tk) => tk[1]);
      const mine = [];
      for (let k = 0; k + 1 < xs.length; k++) {
        const a = xs[k], c = xs[k + 1];
        if (c - a <= GUTTER_GAP * hh) continue;
        const gl = a + 0.35 * hh, gr = c - 0.35 * hh;
        let near = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          if (Math.abs(rights(j) - gl) < 0.5 * hh || Math.abs(lefts(j) - gr) < 0.5 * hh) near++;
        }
        if (near >= GUTTER_EDGES) mine.push((a + c) / 2);
      }
      cuts.push(mine);
    }
    return cuts;
  }

  decodeToks(toks) {
    if (toks.length && typeof toks[0][0] === "string")
      return toks.map((t) => t[0]).join("").split(/\s+/).filter(Boolean).join(" ");
    return this.decodeIds(toks.map((t) => t[0]), this.vocab);
  }

  /** split_line: cut a line where neighbouring chars are > gap line heights apart. */
  splitLine(box, toks, gap, at = []) {
    const hh = Math.max(1, box[3] - box[1]);
    const segs = [[]];
    let prev = null;
    const atSorted = [...at].sort((a, b) => a - b);
    for (const tk of toks) {
      const x = tk[1];
      if (!this.isSpace(tk[0])) {
        const cut = prev !== null && (x - prev > gap * hh || atSorted.some((c) => prev < c && c < x));
        if (cut) segs.push([]);
        prev = x;
      }
      segs[segs.length - 1].push(tk);
    }
    if (segs.length === 1) return [[box.slice(0, 4), this.decodeToks(toks), toks]];
    const out = [];
    for (const sg of segs) {
      const text = this.decodeToks(sg).trim();
      const xs = sg.filter((tk) => !this.isSpace(tk[0])).map((tk) => tk[1]);
      if (text && xs.length) {
        out.push([[
          Math.max(box[0], Math.min(...xs) - 0.35 * hh),
          box[1],
          Math.min(box[2], Math.max(...xs) + 0.35 * hh),
          box[3],
        ], text, sg]);
      }
    }
    return out;
  }

  /** order model forward: returns {order, cls, same, tab, logits} */
  async readingOrder(boxes, texts, W, H) {
    const n = boxes.length / 4;
    if (n < 2 || !this.s.order) {
      return { order: n ? orderXycut(boxes) : [], cls: null, same: null, tab: null, logits: null };
    }
    const f = lineFeatures(boxes, W, H, texts);
    const feeds = {
      feats: new this.ort.Tensor("float32", f, [1, n, N_FEATS]),
      mask: new this.ort.Tensor("bool", new Uint8Array(n).fill(1), [1, n]),
    };
    const outs = await this.s.order.run(feeds);
    const ot = this.s.order.outputNames.map((nm) => outs[nm].data);
    const logits = ot[0];
    if (ot.length >= 3) {
      const cls = ot[1], same = ot[2];
      const tab = ot.length > 3 ? this.unpackTab(ot[3], n) : null;
      const order = decodeBlocksOrder(logits, same, boxes, cls, n);
      return { order, cls, same, tab, logits };
    }
    return { order: decodeOrder(logits, n), cls: null, same: null, tab: null, logits };
  }

  unpackTab(t, n) {
    // [1,3,n,n] or [3,n,n] -> [3][n*n]
    const flat = t.length === 3 * n * n ? t : t.slice(0, 3 * n * n);
    return [flat.subarray(0, n * n), flat.subarray(n * n, 2 * n * n), flat.subarray(2 * n * n)];
  }

  async orientation(gray, w, h) {
    if (!this.s.orient) return 0;
    const x = orientInput(gray, w, h);
    const res = await this.s.orient.run({ x: new this.ort.Tensor("float32", x, [1, 1, ORIENT_SIDE, ORIENT_SIDE]) });
    const p = res[this.s.orient.outputNames[0]].data;
    let k = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
    return p[k] > ORIENT_MIN ? k : 0;
  }

  /** Full OCR of a gray page. Returns {lines:[{box,text,conf}], markdown, blocks, rotation, skew, timings}. */
  async ocrPage(gray, w, h, turn = true) {
    const t = {};
    let t0 = performance.now();
    let k = turn ? await this.orientation(gray, w, h) : 0;
    if (k) ({ data: gray, w, h } = rot90cw(gray, w, h, k));
    let skew = estimateSkew(gray, w, h);
    if (skew) gray = deskew(gray, w, h, skew);
    t.orient_ms = performance.now() - t0;

    t0 = performance.now();
    let boxes = await this.detect(gray, w, h);
    t.det_ms = performance.now() - t0;

    t0 = performance.now();
    const rec = await this.recognize(gray, w, h, boxes);
    t.rec_ms = performance.now() - t0;

    const page = await this.assemble(w, h, boxes, rec, t);
    page.rotation = 90 * k;
    page.skew = Math.round(skew * 100) / 100;
    if (skew) {
      for (const ln of page.lines) ln.box = unrotateBox(ln.box, skew, w, h);
      for (const b of page.blocks) b.bbox = unrotateBox(b.bbox, skew, w, h);
    }
    return page;
  }

  /** boxes [N,5], rec [{text,conf,toks}] -> ordered lines + markdown (pipeline.assemble). */
  async assemble(w, h, boxes, rec, t, { contentOrder = false } = {}) {
    const t0 = performance.now();
    const n0 = boxes.length / 5;
    let bx4 = new Float32Array(n0 * 4);
    for (let i = 0; i < n0; i++) bx4.set(boxes.subarray(5 * i, 5 * i + 4), 4 * i);
    const cuts = n0 ? this.gutterCuts(bx4, rec) : [];
    const nb = [];
    let nrec = [];
    for (let i = 0; i < n0; i++) {
      for (const [bx, tx, tk] of this.splitLine(bx4.subarray(4 * i, 4 * i + 4), rec[i].toks, SPLIT_GAP, cuts[i])) {
        nb.push(...bx); nrec.push({ text: tx, conf: rec[i].conf, toks: tk });
      }
    }
    let boxes4 = Float32Array.from(nb);
    let n = boxes4.length / 4;
    let texts = nrec.map((r) => r.text);
    let { order, cls, same, tab, logits } = await this.readingOrder(boxes4, texts, w, h);

    // table cells: split table lines at CELL_GAP then re-order (pipeline.assemble)
    if (cls && tab) {
      const C = cls.length / n;
      const isTable = [];
      for (let i = 0; i < n; i++) {
        let am = 0;
        for (let c = 1; c < C; c++) if (cls[i * C + c] > cls[i * C + am]) am = c;
        isTable.push(am === BLOCK_CLASSES.indexOf("Table"));
      }
      if (isTable.some(Boolean)) {
        const nb2 = [], nrec2 = [];
        for (let i = 0; i < n; i++) {
          const parts = isTable[i]
            ? this.splitLine(boxes4.subarray(4 * i, 4 * i + 4), nrec[i].toks, CELL_GAP)
            : [[boxes4.subarray(4 * i, 4 * i + 4), nrec[i].text, nrec[i].toks]];
          for (const [bx, tx, tk] of parts) { nb2.push(...bx); nrec2.push({ text: tx, conf: nrec[i].conf, toks: tk }); }
        }
        if (nb2.length / 4 !== n) {
          boxes4 = Float32Array.from(nb2); nrec = nrec2;
          n = boxes4.length / 4; texts = nrec.map((r) => r.text);
          ({ order, cls, same, tab, logits } = await this.readingOrder(boxes4, texts, w, h));
        }
      }
    }

    let note = "order=model";
    if (contentOrder && n > 1) {
      const keys = nrec.map((r) => {
        const ks = r.toks.filter((tk) => tk.length > 2).map((tk) => tk[2]);
        return ks.length ? Math.min(...ks) : null;
      });
      if (keys.every((v) => v !== null)) {
        const pdf = [...Array(n).keys()].sort((a, b) => keys[a] - keys[b]);
        const pb = new Float32Array(n * 4);
        pdf.forEach((v, i) => pb.set(boxes4.subarray(4 * v, 4 * v + 4), 4 * i));
        const pl = logits
          ? (() => { const m = new Float32Array(n * n); pdf.forEach((i2, a) => pdf.forEach((j2, b) => (m[a * n + b] = logits[i2 * n + j2]))); return m; })()
          : null;
        if (pdfOrderCoherent(pb, pl)) { order = pdf; note = "order=pdf"; }
      }
    }
    t.order_ms = performance.now() - t0;
    t.total_ms = Object.values(t).reduce((s, v) => s + v, 0);

    const lines = order.map((i) => ({
      box: [0, 1, 2, 3].map((k) => Math.round(boxes4[4 * i + k] * 10) / 10),
      text: nrec[i].text,
      conf: Math.round(nrec[i].conf * 1e4) / 1e4,
    }));
    let markdown = null, blocks = [];
    if (cls && same) {
      blocks = pageBlocks(boxes4, texts, order, cls, same, n);
      const cells = new Map();
      for (let i = 0; i < n; i++) {
        const parts = this.splitLine(boxes4.subarray(4 * i, 4 * i + 4), nrec[i].toks, CELL_GAP);
        cells.set(i, parts.map(([bx, tx]) => [bx, tx]));
      }
      markdown = pageMarkdown(boxes4, texts, order, cls, same, n, { cells, tab });
    } else if (n < 2) {
      markdown = texts.join("\n");
    }
    return { lines, markdown, blocks, note, timings: t };
  }

  /** Order+layout for externally-supplied lines (text-layer path). */
  async orderAndLayout(boxes4, texts, w, h, { contentOrder = false, contentIdx = null } = {}) {
    const t = {};
    const n = boxes4.length / 4;
    let { order, cls, same, tab, logits } = await this.readingOrder(boxes4, texts, w, h);
    let note = "order=model";
    if (contentOrder && n > 1 && contentIdx) {
      const pdf = [...Array(n).keys()].sort((a, b) => contentIdx[a] - contentIdx[b]);
      const pb = new Float32Array(n * 4);
      pdf.forEach((v, i) => pb.set(boxes4.subarray(4 * v, 4 * v + 4), 4 * i));
      const pl = logits
        ? (() => { const m = new Float32Array(n * n); pdf.forEach((i2, a) => pdf.forEach((j2, b) => (m[a * n + b] = logits[i2 * n + j2]))); return m; })()
        : null;
      if (pdfOrderCoherent(pb, pl)) { order = pdf; note = "order=pdf"; }
    }
    const lines = order.map((i) => ({
      box: [0, 1, 2, 3].map((k) => Math.round(boxes4[4 * i + k] * 10) / 10),
      text: texts[i], conf: 1,
    }));
    let markdown = null, blocks = [];
    if (cls && same) {
      blocks = pageBlocks(boxes4, texts, order, cls, same, n);
      markdown = pageMarkdown(boxes4, texts, order, cls, same, n, { tab });
    } else {
      markdown = lines.map((l) => l.text).join("\n");
    }
    return { lines, markdown, blocks, note, timings: t };
  }
}
