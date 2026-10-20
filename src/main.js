// App orchestration: file queue -> pdf.js (main) + OCR worker (background),
// incremental doc store, live search (regex / fuzzy on WebGPU or JS, semantic
// via an embedding worker), source-page viewer with precise highlights,
// library + dumps (.jsonl, .parquet).
import {
  openPdf, pageTextRuns, textLayerOk, runsToLines, renderGray,
} from "./pdf-extract.js";
import { rgbaToGray } from "./image.js";
import { Store } from "./store.js";
import { compilePattern, literalize } from "./regex.js";
import { searchAll } from "./search.js";
import { GpuSearch, CAP } from "./gpu-search.js";
import { compileFuzzy, fuzzyAll } from "./fuzzy.js";
import { pageRuns, lineRangeRects } from "./highlight.js";
import { Semantic } from "./semantic.js";

const $ = (s) => document.querySelector(s);
const els = {
  q: $("#q"), ci: $("#opt-ci"), ciWrap: $("#ci-wrap"), backend: $("#backend"), qstatus: $("#qstatus"),
  modes: $("#modes"), semPanel: $("#sem-panel"),
  advBtn: $("#adv-btn"), advPop: $("#adv-pop"), optSemantic: $("#opt-semantic"),
  stats: $("#stats"), queue: $("#queue"), results: $("#results"),
  dropzone: $("#dropzone"), fileInput: $("#file-input"),
  mapwrap: $("#mapwrap"), map: $("#map"), maptip: $("#maptip"),
  libTable: $("#lib-table tbody"), libUsage: $("#lib-usage"),
  exportBtn: $("#btn-export"), parquetBtn: $("#btn-parquet"), importBtn: $("#btn-import"), clearBtn: $("#btn-clear"),
  importFile: $("#import-file"),
  viewer: $("#viewer"), vTitle: $("#v-title"), vSub: $("#v-sub"), vBody: $("#v-body"),
  vPage: $("#v-page"), vCanvas: $("#v-canvas"), vHl: $("#v-hl"),
  vPrev: $("#v-prev"), vNext: $("#v-next"), vClose: $("#v-close"),
};

const DPI = 200;

// ---------- OCR worker RPC ----------

class OcrClient {
  constructor() {
    this.worker = new Worker(new URL("./ocr.worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.seq = 0;
    this.ready = false;
    this.worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === "progress") { setStatus(d.msg); return; }
      const p = this.pending.get(d.id);
      if (!p) return;
      this.pending.delete(d.id);
      if (d.type === "error") p.rej(new Error(d.error));
      else p.res(d);
    };
    this.worker.onerror = (e) => {
      const msg = `OCR worker failed: ${e.message || "script load error"} (${(e.filename || "").split("/").pop()}:${e.lineno})`;
      setStatus(msg);
      for (const p of this.pending.values()) p.rej(new Error(msg));
      this.pending.clear();
    };
  }
  call(cmd, payload = {}, transfer = []) {
    const id = ++this.seq;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.worker.postMessage({ cmd, id, ...payload }, transfer);
    });
  }
}

// ---------- app state ----------

const store = new Store();
const ocr = new OcrClient();
const semantic = new Semantic(store);
let gpu = null;
const fileHandles = new Map();   // fileId -> File (this session; OPFS otherwise)
const processing = new Map();    // fileId -> true while parsing
let mode = "regex";              // regex | fuzzy | semantic
let pattern = null, patternSrc = null, patternFlags = null;
let searching = false, searchQueued = false;
let searchMs = 0, lastSearchT = 0, searchTimer = null;
let last = { res: null, order: [], scores: null };   // current results

let semanticAllowed = false;     // advanced option; off by default
try {
  mode = localStorage.getItem("ls-mode") || "regex";
  semanticAllowed = localStorage.getItem("ls-adv-semantic") === "1";
} catch { /* no storage */ }
if (mode === "semantic" && !semanticAllowed) mode = "regex";

// ---------- search ----------

function currentPattern(src, flags) {
  if (pattern && patternSrc === src && patternFlags === flags) return pattern;
  let { tables, error } = compilePattern(src, flags);
  let usedLiteral = false;
  if (error) {
    // plain-word convenience: retry as escaped literal
    const lit = literalize(src);
    const r2 = compilePattern(lit, flags);
    if (!r2.error) { tables = r2.tables; error = null; usedLiteral = true; }
  }
  pattern = { tables, error, usedLiteral };
  patternSrc = src; patternFlags = flags;
  return pattern;
}

function scheduleSearch() {
  if (searching) { searchQueued = true; return; }
  // throttle: during ingestion, docs events fire per page — cap at ~4 runs/sec
  const wait = 250 - (performance.now() - lastSearchT);
  if (wait > 0) {
    if (!searchTimer)
      searchTimer = setTimeout(() => { searchTimer = null; scheduleSearch(); }, wait);
    return;
  }
  searching = true;
  lastSearchT = performance.now();
  runSearch().catch((e) => { console.error(e); setStatus(String(e).slice(0, 140)); searching = false; });
}

function syncGpuDocs() {
  const nDocs = store.docs.length;
  if (store.dirty || gpu.nDocs !== nDocs) {
    gpu.setDocs(store.cells.subarray(0, store.cellUsed), store.docLen.subarray(0, nDocs));
    store.dirty = false;
  }
}

function gpuFailed(e) {
  console.warn("gpu search failed, falling back to js", e);
  gpu = null;
  els.backend.textContent = "JS";
  els.backend.classList.add("off");
}

async function runRegex(src, flags) {
  const p = currentPattern(src, flags);
  if (p.error) return { error: p.error };
  const cells = () => [store.cells.subarray(0, store.cellUsed), store.docLen.subarray(0, store.docs.length)];
  let res;
  if (gpu) {
    try {
      if (gpu.tab !== p.tables) gpu.setPattern(p.tables);
      syncGpuDocs();
      res = await gpu.run();
    } catch (e) { gpuFailed(e); }
  }
  if (!res) { res = searchAll(p.tables, ...cells()); store.dirty = false; }
  const order = [];
  res.forEach((s, d) => { if (s) order.push(d); });
  return { res, order, note: p.usedLiteral ? "literal" : "" };
}

async function runFuzzy(src) {
  const f = compileFuzzy(src);
  if (f.error) return { error: f.error };
  let res;
  if (gpu) {
    try { syncGpuDocs(); res = await gpu.runFuzzy(f); } catch (e) { console.warn("gpu fuzzy failed, using js", e); }
  }
  if (!res) {
    res = fuzzyAll(f, store.cells.subarray(0, store.cellUsed), store.docLen.subarray(0, store.docs.length), CAP);
    store.dirty = false;
  }
  // rank: fewest edits first, then document order
  const best = new Map();
  res.forEach((s, d) => { if (s) best.set(d, Math.min(...s.map((x) => x[2]))); });
  const pageBest = new Map();
  for (const [d, e] of best) {
    const k = pageKey(store.docs[d]);
    pageBest.set(k, Math.min(pageBest.get(k) ?? 99, e));
  }
  const order = [...best.keys()].sort((a, b) =>
    (pageBest.get(pageKey(store.docs[a])) - pageBest.get(pageKey(store.docs[b]))) || a - b);
  return { res, order, note: `≤${f.k} edit${f.k === 1 ? "" : "s"}` };
}

async function runSemantic(src) {
  if (semantic.state !== "ready") { renderSemPanel(); return { res: [], order: [], silent: true }; }
  const hits = await semantic.search(src, 20);
  const res = new Array(store.docs.length).fill(null);
  const order = [], scores = new Map();
  for (const h of hits) {
    for (const d of h.docs) {
      if (res[d]) continue;
      res[d] = [[0, Math.min(store.docs[d].fullLen, CAP)]];
      order.push(d); scores.set(d, h.score);
    }
  }
  return { res, order, scores, passages: hits, note: `${hits.length} closest passages` };
}

async function runSearch() {
  const src = els.q.value.trim();
  const flags = els.ci.checked ? "i" : "";
  renderSemPanel();
  if (!src) {
    els.results.replaceChildren();
    els.qstatus.textContent = "";
    last = { res: null, order: [], scores: null };
    pageHitMap.clear(); scheduleMapRender();
    searching = false;
    return;
  }
  const t0 = performance.now();
  const r = mode === "fuzzy" ? await runFuzzy(src)
    : mode === "semantic" ? await runSemantic(src)
    : await runRegex(src, flags);
  if (r.error) {
    els.qstatus.textContent = r.error;
    searching = false;
    return;
  }
  searchMs = performance.now() - t0;
  last = { res: r.res, order: r.order, scores: r.scores || null };
  renderResults(r);
  searching = false;
  if (searchQueued || (store.dirty && mode !== "semantic")) { searchQueued = false; scheduleSearch(); }
}

function cpToStrIdx(text) {
  // cell index (code point) -> string index, for the doc's first CAP cps
  const map = new Uint32Array(CAP + 1);
  let ci = 0, si = 0;
  for (const ch of text) {
    if (ci >= CAP) break;
    map[ci] = si; si += ch.length; ci++;
  }
  for (let k = ci; k <= CAP; k++) map[k] = si;
  return map;
}

/** Match spans of doc d as string ranges [a,b,errors?]. */
function strRanges(d) {
  const doc = store.docs[d], spans = last.res && last.res[d];
  if (!spans) return [];
  const map = cpToStrIdx(doc.text);
  return spans.map(([s, e, err]) => [map[Math.min(s, CAP)], map[Math.min(e, CAP)], err]);
}

function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const pageKey = (doc) => doc.fileId + ":" + doc.page;

function renderResults(r) {
  const R = els.results;
  const { order } = r;
  let total = 0;
  for (const d of order) total += last.res[d].length;
  const head = mode === "semantic"
    ? r.note
    : `${total.toLocaleString()} match${total === 1 ? "" : "es"} in ${order.length.toLocaleString()} line${order.length === 1 ? "" : "s"}${r.note ? " · " + r.note : ""}`;
  els.qstatus.textContent = `${head} · ${searchMs.toFixed(1)}ms${processing.size ? " · parsing…" : ""}`;
  const LIMIT = 400;
  const frag = document.createDocumentFragment();
  let lastKey = "";
  if (r.passages) {
    for (const h of r.passages) {
      if (!h.docs.length) continue;
      const doc = store.docs[h.docs[0]], end = store.docs[h.docs[h.docs.length - 1]];
      const row = document.createElement("div");
      row.className = "res passage";
      row.dataset.doc = h.docs[0];
      const head = document.createElement("div");
      head.className = "p-head";
      head.innerHTML = `<span class="p-file"></span><span class="where">p${doc.page + 1} · L${doc.lineIdx + 1}–${end.lineIdx + 1}</span><span class="score">${Math.round(h.score * 100)}%</span>`;
      head.querySelector(".p-file").textContent = doc.fileName;
      const body = document.createElement("div");
      body.className = "p-text";
      body.textContent = h.docs.map((d) => store.docs[d].text).join(" ").replace(/(\w)- (\w)/g, "$1$2");
      row.append(head, body);
      frag.appendChild(row);
    }
  }
  const lineOrder = r.passages ? [] : order;   // passages render above instead
  lineOrder.slice(0, LIMIT).forEach((d) => {
    const doc = store.docs[d];
    const key = pageKey(doc);
    if (key !== lastKey) {
      lastKey = key;
      const h = document.createElement("div");
      h.className = "res-head";
      h.textContent = `${doc.fileName} — page ${doc.page + 1}`;
      if (last.scores) {
        const sc = document.createElement("span");
        sc.className = "score";
        sc.textContent = `${Math.round(last.scores.get(d) * 100)}% similar`;
        h.appendChild(sc);
      }
      frag.appendChild(h);
    }
    const row = document.createElement("div");
    row.className = "res";
    row.dataset.doc = d;
    const { before, after } = store.context(d, 1);
    const ctxB = before.map((x) => `<div class="ctx">${escHtml(x.text)}</div>`).join("");
    const ctxA = after.map((x) => `<div class="ctx">${escHtml(x.text)}</div>`).join("");
    let html = "", cur = 0;
    if (mode === "semantic") html = escHtml(doc.text);
    else {
      for (const [a, b, err] of strRanges(d)) {
        if (b <= a || a < cur) continue;
        const cls = err ? ` class="fz" title="${err} edit${err > 1 ? "s" : ""}"` : "";
        html += escHtml(doc.text.slice(cur, a)) + `<mark${cls}>` + escHtml(doc.text.slice(a, b)) + "</mark>";
        cur = b;
      }
      html += escHtml(doc.text.slice(cur));
    }
    if (doc.fullLen > CAP) html += " …";
    row.innerHTML = `<div class="where">p${doc.page + 1} · L${doc.lineIdx + 1}</div><div>${mode === "semantic" ? "" : ctxB}<div class="line">${html}</div>${mode === "semantic" ? "" : ctxA}</div>`;
    frag.appendChild(row);
  });
  if (lineOrder.length > LIMIT) {
    const more = document.createElement("div");
    more.className = "nores";
    more.textContent = `… and ${(lineOrder.length - LIMIT).toLocaleString()} more lines — refine the search`;
    frag.appendChild(more);
  }
  if (!order.length && !r.silent && !r.passages?.length) {
    const none = document.createElement("div");
    none.className = "nores";
    none.textContent = processing.size ? "No matches yet — still parsing" : "No matches";
    frag.appendChild(none);
  }
  R.replaceChildren(frag);

  // aggregate hits per page (map + viewer), in result order
  pageHitMap.clear();
  for (const d of order) {
    const k = pageKey(store.docs[d]);
    const arr = pageHitMap.get(k);
    if (arr) { if (arr.length < 64) arr.push(d); } else pageHitMap.set(k, [d]);
  }
  scheduleMapRender();
}

// ---------- semantic panel ----------

function renderSemPanel() {
  const P = els.semPanel;
  if (mode !== "semantic") { P.classList.add("hidden"); return; }
  const s = semantic;
  const ready = s.state === "ready" && !s.indexing;
  P.classList.toggle("hidden", ready);
  if (ready) return;
  const btn = (label, fn) => {
    const b = document.createElement("button");
    b.className = "btn primary"; b.textContent = label; b.onclick = fn;
    return b;
  };
  const head = document.createElement("div");
  head.className = "sem-title";
  const body = document.createElement("div");
  body.className = "sem-body";
  const parts = [head, body];
  if (s.state === "off" || s.state === "error") {
    head.textContent = "Search by meaning";
    body.textContent = `Runs ${s.modelName} on this device. The model (${s.modelSize}) downloads once from Hugging Face and is cached; your documents never leave the browser.`;
    if (s.state === "error") {
      const e = document.createElement("div");
      e.className = "sem-err"; e.textContent = s.error;
      parts.push(e);
    }
    parts.push(btn(s.state === "error" ? "Try again" : "Enable semantic search", () => semantic.enable()));
  } else {
    head.textContent = s.state === "loading" ? "Loading model…" : "Indexing passages…";
    body.textContent = s.progressText || "";
    const bar = document.createElement("div");
    bar.className = "bar";
    const i = document.createElement("i");
    i.style.width = `${Math.round(100 * (s.progress || 0))}%`;
    bar.appendChild(i);
    parts.push(bar);
  }
  P.replaceChildren(...parts);
}

semantic.addEventListener("change", () => {
  renderSemPanel();
  if (mode === "semantic" && semantic.state === "ready" && !semantic.indexing && els.q.value.trim()) scheduleSearch();
});

// ---------- source-page viewer ----------

const viewer = { pages: [], idx: 0, focus: -1, pdf: null, pdfFileId: null, token: 0 };

async function sourceFile(fileId) {
  let f = fileHandles.get(fileId);
  if (!f) {
    const raw = await store.getOriginal(fileId);
    const rec = store.files.get(fileId);
    if (raw && rec) {
      // OPFS names entries by id; restore the real name/type for sniffing
      const type = /\.pdf$/i.test(rec.name) ? "application/pdf" : raw.type;
      f = new File([raw], rec.name, { type });
      fileHandles.set(fileId, f);
    }
  }
  return f;
}

async function pdfFor(fileId, file) {
  if (viewer.pdfFileId === fileId && viewer.pdf) return viewer.pdf;
  if (viewer.pdf) viewer.pdf.destroy();
  viewer.pdf = await openPdf(await file.arrayBuffer());
  viewer.pdfFileId = fileId;
  return viewer.pdf;
}

/** Open the viewer on a page key; `focusDoc` is scrolled to and emphasised. */
function openViewer(key, focusDoc = -1) {
  const keys = [...pageHitMap.keys()];
  viewer.pages = keys.length ? keys : [key];
  viewer.idx = Math.max(0, viewer.pages.indexOf(key));
  if (!viewer.pages.includes(key)) { viewer.pages = [key]; viewer.idx = 0; }
  viewer.focus = focusDoc;
  els.viewer.classList.remove("hidden");
  document.body.classList.add("noscroll");
  showViewerPage();
}

function closeViewer() {
  els.viewer.classList.add("hidden");
  document.body.classList.remove("noscroll");
  viewer.token++;
}

function stepViewer(dir) {
  if (viewer.pages.length < 2) return;
  viewer.idx = (viewer.idx + dir + viewer.pages.length) % viewer.pages.length;
  viewer.focus = -1;
  showViewerPage();
}

async function showViewerPage() {
  const token = ++viewer.token;
  const key = viewer.pages[viewer.idx];
  const [fileId, pageStr] = [key.slice(0, key.lastIndexOf(":")), key.slice(key.lastIndexOf(":") + 1)];
  const pageNo = +pageStr;
  const f = store.files.get(fileId);
  const hits = pageHitMap.get(key) || [];
  els.vTitle.textContent = f ? f.name : "?";
  const nHits = hits.reduce((s, d) => s + (last.res?.[d]?.length || 0), 0);
  els.vSub.textContent = `Page ${pageNo + 1}${f?.pageCount ? ` of ${f.pageCount}` : ""}` +
    (hits.length ? ` · ${nHits} ${mode === "semantic" ? "line" : "match"}${nHits === 1 ? "" : mode === "semantic" ? "s" : "es"}` : "") +
    (viewer.pages.length > 1 ? ` · result page ${viewer.idx + 1}/${viewer.pages.length}` : "");
  els.vPrev.disabled = els.vNext.disabled = viewer.pages.length < 2;
  els.vPage.classList.add("loading");

  const file = await sourceFile(fileId);
  if (token !== viewer.token) return;
  if (!file) {
    els.vHl.replaceChildren();
    els.vCanvas.width = els.vCanvas.height = 0;
    els.vPage.classList.remove("loading");
    els.vPage.dataset.missing = "The original file isn't stored in this browser (imported from a dump). Drop it in again to see pages.";
    return;
  }
  delete els.vPage.dataset.missing;

  const dpr = devicePixelRatio || 1;
  const cssW = Math.min(920, els.vBody.clientWidth - 48);
  let W200, H200, runs = null;
  const cv = els.vCanvas, ctx = cv.getContext("2d");
  try {
    if (/\.pdf$/i.test(file.name) || file.type === "application/pdf") {
      const pdf = await pdfFor(fileId, file);
      const page = await pdf.getPage(pageNo + 1);
      const vp200 = page.getViewport({ scale: DPI / 72 });
      W200 = vp200.width; H200 = vp200.height;
      const scale = (cssW * dpr) / page.getViewport({ scale: 1 }).width;
      const vp = page.getViewport({ scale });
      const off = document.createElement("canvas");
      off.width = Math.ceil(vp.width); off.height = Math.ceil(vp.height);
      const octx = off.getContext("2d");
      octx.fillStyle = "#fff"; octx.fillRect(0, 0, off.width, off.height);
      const [, r] = await Promise.all([
        page.render({ canvasContext: octx, viewport: vp }).promise,
        hits.length ? pageRuns(page, vp200) : null,
      ]);
      runs = r;
      if (token !== viewer.token) return;
      cv.width = off.width; cv.height = off.height;
      ctx.drawImage(off, 0, 0);
    } else {
      const bmp = await createImageBitmap(file);
      W200 = bmp.width; H200 = bmp.height;   // OCR boxes are in image px
      cv.width = Math.round(cssW * dpr); cv.height = Math.round(cv.width * bmp.height / bmp.width);
      ctx.drawImage(bmp, 0, 0, cv.width, cv.height);
    }
  } catch (e) {
    console.warn("viewer render failed", e);
    els.vPage.classList.remove("loading");
    return;
  }
  cv.style.width = cssW + "px";
  cv.style.height = (cssW * cv.height / cv.width) + "px";

  // highlights as % of the page so they track the canvas size
  const frag = document.createDocumentFragment();
  let focusEl = null;
  for (const d of hits) {
    const doc = store.docs[d];
    const ranges = mode === "semantic" ? [[0, doc.text.length]] : strRanges(d);
    const rects = lineRangeRects(runs, doc, ranges);
    for (const { rect, exact } of rects) {
      const el = document.createElement("div");
      el.className = "hl" + (exact ? "" : " approx") + (mode === "semantic" ? " passage" : "") + (d === viewer.focus ? " active" : "");
      const pad = 0.12 * (rect[3] - rect[1]);
      el.style.left = (100 * (rect[0] - pad / 2) / W200) + "%";
      el.style.top = (100 * (rect[1] - pad / 2) / H200) + "%";
      el.style.width = (100 * (rect[2] - rect[0] + pad) / W200) + "%";
      el.style.height = (100 * (rect[3] - rect[1] + pad) / H200) + "%";
      el.title = doc.text;
      frag.appendChild(el);
      if (!focusEl && (d === viewer.focus || viewer.focus < 0)) focusEl = el;
    }
  }
  els.vHl.replaceChildren(frag);
  els.vPage.classList.remove("loading");
  if (focusEl) requestAnimationFrame(() => focusEl.scrollIntoView({ block: "center", behavior: "smooth" }));
  else els.vBody.scrollTop = 0;
}

els.vPrev.onclick = () => stepViewer(-1);
els.vNext.onclick = () => stepViewer(1);
els.vClose.onclick = closeViewer;
els.viewer.addEventListener("click", (e) => { if (e.target === els.viewer || e.target === els.vBody) closeViewer(); });

// ---------- page map (grid: the book, cell = page) ----------

const pageHitMap = new Map();   // "fileId:page" -> [docIdx, ...] in result order
let pageIndex = [];             // cell i -> {fileId, page, key}
let pageIndexStale = true;
let mapRaf = 0;
let nextHitPos = 0;             // for Enter = next match

function rebuildPageIndex() {
  pageIndex = [];
  for (const f of store.files.values())
    for (let p = 0; p < (f.pageCount || 0); p++)
      pageIndex.push({ fileId: f.id, page: p, key: f.id + ":" + p });
  pageIndexStale = false;
}

const CELL = 11, GAP = 3;
let mapColors = null;
function readMapColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return { hit: v("--map-hit"), parsed: v("--map-parsed"), pending: v("--map-pending") };
}

function scheduleMapRender() {
  if (mapRaf) return;
  mapRaf = requestAnimationFrame(() => { mapRaf = 0; renderMap(); });
}

function renderMap() {
  if (pageIndexStale) rebuildPageIndex();
  const total = pageIndex.length;
  const wrap = els.mapwrap, cv = els.map;
  if (!total) { wrap.classList.add("hidden"); return; }
  wrap.classList.remove("hidden");

  const w = wrap.clientWidth - 32; // padding
  const cols = Math.max(1, Math.floor(w / (CELL + GAP)));
  const rows = Math.ceil(total / cols);
  const dpr = devicePixelRatio || 1;
  const W = cols * (CELL + GAP) - GAP, H = rows * (CELL + GAP) - GAP;
  if (cv.width !== W * dpr || cv.height !== H * dpr) {
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + "px"; cv.style.height = H + "px";
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const parsed = store.pageLines;
  const col = mapColors ||= readMapColors();
  for (let i = 0; i < total; i++) {
    const c = pageIndex[i];
    const x = (i % cols) * (CELL + GAP), y = ((i / cols) | 0) * (CELL + GAP);
    ctx.beginPath();
    ctx.roundRect(x, y, CELL, CELL, 2.5);
    if (pageHitMap.has(c.key)) ctx.fillStyle = col.hit;
    else if (parsed.has(c.key)) ctx.fillStyle = col.parsed;
    else ctx.fillStyle = col.pending;   // still parsing
    ctx.fill();
  }
}

function cellAt(ev) {
  const r = els.map.getBoundingClientRect();
  const cols = Math.max(1, Math.floor((r.width) / (CELL + GAP)));
  const cx = Math.floor((ev.clientX - r.left) / (CELL + GAP));
  const cy = Math.floor((ev.clientY - r.top) / (CELL + GAP));
  // ignore gap pixels
  if ((ev.clientX - r.left) % (CELL + GAP) > CELL || (ev.clientY - r.top) % (CELL + GAP) > CELL) return -1;
  const i = cy * cols + cx;
  return i >= 0 && i < pageIndex.length ? i : -1;
}

els.map.addEventListener("mousemove", (e) => {
  const i = cellAt(e);
  if (i < 0) { els.maptip.classList.add("hidden"); return; }
  const c = pageIndex[i];
  const f = store.files.get(c.fileId);
  const hits = pageHitMap.get(c.key);
  const tip = els.maptip;
  tip.innerHTML = "";
  const head = document.createElement("div");
  head.className = "tip-head";
  head.textContent = `${f ? f.name : "?"} — page ${c.page + 1}`;
  tip.appendChild(head);
  if (hits && hits.length) {
    const doc = store.docs[hits[0]];
    const line = document.createElement("div");
    line.className = "tip-line";
    line.textContent = doc.text.slice(0, 120);
    tip.appendChild(line);
    if (hits.length > 1) {
      const more = document.createElement("div");
      more.className = "tip-line";
      more.textContent = `+${hits.length - 1} more line${hits.length > 2 ? "s" : ""}`;
      tip.appendChild(more);
    }
  } else {
    const s = document.createElement("div");
    s.className = "tip-line";
    s.textContent = store.pageLines.has(c.key) ? "no match" : "not parsed yet";
    tip.appendChild(s);
  }
  tip.classList.remove("hidden");
  tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 12) + "px";
  tip.style.top = Math.min(e.clientY + 16, innerHeight - tip.offsetHeight - 12) + "px";
});
els.map.addEventListener("mouseleave", () => els.maptip.classList.add("hidden"));
els.map.addEventListener("click", (e) => {
  const i = cellAt(e);
  if (i < 0) return;
  els.maptip.classList.add("hidden");
  openViewer(pageIndex[i].key);
});

// Enter in the query box = open the next matching page
els.q.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || !pageHitMap.size) return;
  const keys = [...pageHitMap.keys()];
  openViewer(keys[nextHitPos++ % keys.length]);
});

addEventListener("resize", () => { pageIndexStale = true; scheduleMapRender(); });

// ---------- file processing ----------

async function processPdf(file, fileId, qel) {
  const data = await file.arrayBuffer();
  const doc = await openPdf(data);
  const n = doc.numPages;
  store.addFile({ id: fileId, name: file.name, size: file.size, pageCount: n, addedAt: Date.now(), mode: "auto", status: "parsing", lines: 0, note: "" });
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let nLines = 0, nText = 0, nOcr = 0;
  const setProgress = (i) => {
    qel.querySelector(".bar i").style.width = `${(100 * i) / n}%`;
    qel.querySelector(".qmeta").textContent =
      `${i}/${n} pages · ${nText} text · ${nOcr} ocr · ${nLines} lines`;
  };
  for (let i = 1; i <= n; i++) {
    let page = null;
    try {
      page = await doc.getPage(i);
    } catch (e) {
      console.error(`p${i} getPage failed:`, e);
      store.addPage(fileId, i - 1, []);
      setProgress(i);
      continue;
    }
    let done = false;
    const vp = page.getViewport({ scale: DPI / 72 });
    try {
      const tl = await pageTextRuns(page, vp);
      if (textLayerOk(tl)) {
        const { boxes, texts, contentIdx } = runsToLines(tl.runs);
        if (texts.length) {
          const r = await ocr.call("order", {
            boxes: Array.from(boxes), texts,
            w: vp.width, h: vp.height, contentIdx,
          });
          nLines += r.lines.length; nText++;
          store.addPage(fileId, i - 1, r.lines);
          done = true;
        }
      }
    } catch (e) {
      console.warn(`p${i} text layer failed:`, e);
    }
    if (!done) {
      try {
        const { gray, w, h } = await renderGray(page, DPI / 72, canvas, ctx);
        const r = await ocr.call("ocr", { gray: gray.buffer, w, h }, [gray.buffer]);
        nLines += r.lines.length; nOcr++;
        store.addPage(fileId, i - 1, r.lines);
      } catch (e) {
        console.error(`p${i} ocr failed:`, e);
        store.addPage(fileId, i - 1, []);
      }
    }
    setProgress(i);
    page.cleanup();
    if (i % 8 === 0) await new Promise((r) => setTimeout(r, 0)); // yield to UI
  }
  doc.destroy();
  store.updateFile(fileId, { status: "done", lines: nLines });
}

async function processImage(file, fileId, qel) {
  const bmp = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = bmp.width; canvas.height = bmp.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const img = ctx.getImageData(0, 0, bmp.width, bmp.height);
  const gray = rgbaToGray(img.data, bmp.width, bmp.height);
  store.addFile({ id: fileId, name: file.name, size: file.size, pageCount: 1, addedAt: Date.now(), mode: "ocr", status: "parsing", lines: 0, note: "image" });
  qel.querySelector(".qmeta").textContent = "image · ocr…";
  const r = await ocr.call("ocr", { gray: gray.buffer, w: bmp.width, h: bmp.height }, [gray.buffer]);
  store.addPage(fileId, 0, r.lines);
  qel.querySelector(".bar i").style.width = "100%";
  qel.querySelector(".qmeta").textContent = `image · ${r.lines.length} lines`;
  store.updateFile(fileId, { status: "done", lines: r.lines.length });
}

async function processFile(file) {
  const qel = document.createElement("div");
  qel.className = "qitem";
  qel.innerHTML = `<div class="name"></div><div class="qmeta">queued</div><div class="bar"><i></i></div>`;
  qel.querySelector(".name").textContent = file.name;
  els.queue.appendChild(qel);
  const fileId = crypto.randomUUID();
  fileHandles.set(fileId, file);
  processing.set(fileId, true);
  updateStats();
  store.saveOriginal(fileId, file);
  try {
    if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
      await processPdf(file, fileId, qel);
    } else {
      await processImage(file, fileId, qel);
    }
  } catch (e) {
    qel.classList.add("err");
    qel.querySelector(".qmeta").textContent = String(e).slice(0, 120);
    store.updateFile(fileId, { status: "error", note: String(e).slice(0, 200) });
  }
  processing.delete(fileId);
  setTimeout(() => qel.remove(), 4000);
  semantic.onFileDone(fileId);
  scheduleSearch();
}

// ---------- library ----------

function renderLibrary() {
  const tb = els.libTable;
  const rows = [...store.files.values()].sort((a, b) => b.addedAt - a.addedAt);
  const frag = document.createDocumentFragment();
  for (const f of rows) {
    const tr = document.createElement("tr");
    const pagesDone = [...store.pageLines.keys()].filter((k) => k.startsWith(f.id + ":")).length;
    const cls = f.status === "done" ? "status-ok" : f.status === "error" ? "status-err" : "status-run";
    tr.innerHTML = `
      <td class="file"></td>
      <td class="num">${pagesDone}/${f.pageCount || "?"}</td>
      <td class="num">${(f.lines || 0).toLocaleString()}</td>
      <td>${f.mode || ""}</td>
      <td class="date">${new Date(f.addedAt).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</td>
      <td><span class="pill ${cls}"></span></td>
      <td class="row-actions"><button data-del="${f.id}">delete</button></td>`;
    tr.querySelector(".file").textContent = f.name;
    tr.querySelector(".file").title = f.name;
    tr.querySelector(".pill").textContent = f.status + (f.note ? " · " + f.note : "");
    frag.appendChild(tr);
  }
  tb.replaceChildren(frag);
  navigator.storage?.estimate?.().then((e) => {
    els.libUsage.textContent =
      `Documents persist in this browser${e?.usage ? ` (using ${(e.usage / 1048576).toFixed(1)} MB)` : ""}. Export to keep a copy, move machines, or analyze the text elsewhere.`;
  });
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");

async function busy(btn, label, fn) {
  const prev = btn.textContent;
  btn.disabled = true;
  btn.textContent = label;
  try { await fn(); } catch (e) { setStatus(String(e).slice(0, 140)); console.error(e); } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
}

const exportDump = () => busy(els.exportBtn, "Exporting…", async () => {
  const text = await store.exportDump();
  download(new Blob([text], { type: "application/x-ndjson" }), `localscrape-${stamp()}.jsonl`);
});

// one row per line; boxes in PDF points (72/in), top-left origin
const exportParquet = () => busy(els.parquetBtn, "Exporting…", async () => {
  const { parquetWriteBuffer } = await import("../vendor/hyparquet-writer.min.mjs");
  const docs = store.docs;
  const pt = 72 / DPI;
  const col = (f) => docs.map(f);
  const box = (i) => (d) => (d.box ? Math.round(d.box[i] * pt * 100) / 100 : null);
  const buf = parquetWriteBuffer({
    columnData: [
      { name: "file", data: col((d) => d.fileName), type: "STRING" },
      { name: "page", data: col((d) => d.page + 1), type: "INT32" },
      { name: "line", data: col((d) => d.lineIdx + 1), type: "INT32" },
      { name: "text", data: col((d) => d.text), type: "STRING" },
      { name: "x0", data: col(box(0)), type: "DOUBLE" },
      { name: "y0", data: col(box(1)), type: "DOUBLE" },
      { name: "x1", data: col(box(2)), type: "DOUBLE" },
      { name: "y1", data: col(box(3)), type: "DOUBLE" },
    ],
    kvMetadata: [{ key: "created_by_app", value: "localscrape" }, { key: "box_units", value: "pdf points, top-left origin" }],
  });
  download(new Blob([buf], { type: "application/vnd.apache.parquet" }), `localscrape-${stamp()}.parquet`);
});

const importDump = (file) => busy(els.importBtn, "Importing…", async () => {
  const n = await store.importDump(await file.text());
  setStatus(`imported ${n} file(s) from dump`);
  scheduleSearch();
});

// ---------- chrome ----------

function setStatus(s) { els.qstatus.textContent = s; }

function updateStats() {
  const s = store.stats();
  document.body.classList.toggle("has-docs", s.files > 0 || processing.size > 0);
  els.stats.textContent =
    `${s.files} file${s.files === 1 ? "" : "s"} · ${s.pages.toLocaleString()} pages · ${s.docs.toLocaleString()} lines`;
}

function setSemanticAllowed(on) {
  semanticAllowed = on;
  try { on ? localStorage.setItem("ls-adv-semantic", "1") : localStorage.removeItem("ls-adv-semantic"); } catch { /* no storage */ }
  els.optSemantic.checked = on;
  els.modes.querySelector('[data-mode="semantic"]').classList.toggle("hidden", !on);
  if (on) setMode("semantic");
  else {
    semantic.disable();
    if (mode === "semantic") setMode("regex");
  }
}

function setMode(m) {
  if (m === "semantic" && !semanticAllowed) m = "regex";
  mode = m;
  try { localStorage.setItem("ls-mode", m); } catch { /* no storage */ }
  els.modes.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  els.ciWrap.classList.toggle("hidden", m !== "regex");
  els.q.placeholder = {
    regex: "Search with words or regex, e.g. invoice|receipt",
    fuzzy: "Typo- and OCR-tolerant, e.g. modernist planing",
    semantic: "Describe what you're looking for, e.g. why planners simplify cities",
  }[m];
  scheduleSearch();
}

// ---------- wiring ----------

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $("#view-search").classList.toggle("hidden", name !== "search");
  $("#view-library").classList.toggle("hidden", name !== "library");
  if (name === "library") renderLibrary();
}

async function boot() {
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
  els.modes.querySelectorAll("button").forEach((b) => (b.onclick = () => { setMode(b.dataset.mode); els.q.focus(); }));
  els.optSemantic.checked = semanticAllowed;
  els.modes.querySelector('[data-mode="semantic"]').classList.toggle("hidden", !semanticAllowed);
  setMode(mode);
  const toggleAdv = (open) => {
    els.advPop.classList.toggle("hidden", !open);
    els.advBtn.setAttribute("aria-expanded", String(open));
  };
  els.advBtn.onclick = (e) => { e.stopPropagation(); toggleAdv(els.advPop.classList.contains("hidden")); };
  els.optSemantic.onchange = () => setSemanticAllowed(els.optSemantic.checked);
  document.addEventListener("click", (e) => { if (!e.target.closest(".adv")) toggleAdv(false); });

  const dz = els.dropzone;
  const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
  ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { prevent(e); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => { prevent(e); if (ev === "drop" || e.target === document) dz.classList.remove("drag"); }));
  document.addEventListener("drop", (e) => { [...e.dataTransfer.files].forEach(processFile); });
  dz.onclick = () => els.fileInput.click();
  els.fileInput.onchange = () => { [...els.fileInput.files].forEach(processFile); els.fileInput.value = ""; };

  document.addEventListener("keydown", (e) => {
    const open = !els.viewer.classList.contains("hidden");
    if (e.key === "Escape" && !els.advPop.classList.contains("hidden")) { els.advPop.classList.add("hidden"); els.advBtn.setAttribute("aria-expanded", "false"); return; }
    if (open) {
      if (e.key === "Escape") closeViewer();
      else if (e.key === "ArrowRight" || e.key === "j") stepViewer(1);
      else if (e.key === "ArrowLeft" || e.key === "k") stepViewer(-1);
      else return;
      e.preventDefault();
      return;
    }
    if (e.key === "Escape" && document.activeElement === els.q) els.q.blur();
    if (e.key === "/" && !/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName)) {
      e.preventDefault();
      switchTab("search");
      els.q.focus(); els.q.select();
    }
  });

  let debounce = null;
  els.q.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(scheduleSearch, mode === "semantic" ? 350 : 80);
  });
  els.ci.addEventListener("change", scheduleSearch);
  els.results.addEventListener("click", (e) => {
    const row = e.target.closest(".res");
    if (row) openViewer(pageKey(store.docs[+row.dataset.doc]), +row.dataset.doc);
  });
  els.libTable.addEventListener("click", (e) => {
    const id = e.target.dataset?.del;
    if (id && confirm("Delete this file and its extracted text?")) {
      fileHandles.delete(id);
      store.deleteFile(id).then(() => { semantic.onFileDeleted(id); renderLibrary(); scheduleSearch(); });
    }
  });
  els.exportBtn.onclick = exportDump;
  els.parquetBtn.onclick = exportParquet;
  els.importBtn.onclick = () => els.importFile.click();
  els.importFile.onchange = () => { if (els.importFile.files[0]) importDump(els.importFile.files[0]); els.importFile.value = ""; };
  els.clearBtn.onclick = () => {
    if (confirm("Delete ALL stored files and extracted text?")) { fileHandles.clear(); store.clearAll(); semantic.clear(); }
  };

  store.addEventListener("docs", () => { updateStats(); if (els.q.value.trim() && mode !== "semantic") scheduleSearch(); scheduleMapRender(); });
  store.addEventListener("files", () => { pageIndexStale = true; updateStats(); if (!$("#view-library").classList.contains("hidden")) renderLibrary(); scheduleMapRender(); });

  setStatus("loading library…");
  await store.init();
  window.__store = store; // debug: inspect docs/pages in devtools
  window.__semantic = semantic;
  updateStats();
  if (semanticAllowed) semantic.init();

  setStatus("initializing search backend…");
  gpu = await GpuSearch.create();
  els.backend.textContent = gpu ? "WebGPU" : "JS";
  els.backend.classList.toggle("off", !gpu);

  setStatus("warming OCR engine…");
  try {
    const r = await ocr.call("init", {
      modelsBase: new URL("./", location.href).href,
      webgpu: new URLSearchParams(location.search).has("gpu"),
    });
    setStatus(`ready — ${r.eps.join("+")} · vocab ${r.vocab}`);
    setTimeout(() => { if (!els.q.value.trim()) setStatus(""); }, 4000);
  } catch (e) {
    setStatus("OCR engine failed to load: " + String(e).slice(0, 100));
  }
}

boot();
