// Semantic search client: chunks each page's lines into short passages,
// embeds them in embed.worker.js, keeps vectors in IndexedDB, and ranks
// passages by cosine similarity to the query.
const DIMS = 256;
const DB = "localscrape-vec", VER = 1;
const BATCH = 8;
const PASSAGE_CHARS = 420;   // ~80 words, well under the 256-token cap

function openDb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, VER);
    r.onupgradeneeded = () => {
      const s = r.result.createObjectStore("pages", { keyPath: ["fileId", "page"] });
      s.createIndex("fileId", "fileId");
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
const req = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

/** Group a page's lines into passages: [[lineIdx...], text]. */
function passages(lines) {
  const out = [];
  let cur = [], text = "";
  lines.forEach((t, i) => {
    cur.push(i);
    text += (text ? " " : "") + t;
    if (text.length >= PASSAGE_CHARS) { out.push([cur, text]); cur = []; text = ""; }
  });
  if (cur.length) {
    // fold a short tail into the previous passage
    if (out.length && text.length < PASSAGE_CHARS / 3) {
      const p = out[out.length - 1];
      p[0].push(...cur); p[1] += " " + text;
    } else out.push([cur, text]);
  }
  return out;
}

export class Semantic extends EventTarget {
  constructor(store) {
    super();
    this.store = store;
    this.state = "off";          // off | loading | ready | error
    this.indexing = false;
    this.progress = 0; this.progressText = ""; this.error = "";
    this.modelName = "EmbeddingGemma-300M";
    this.modelSize = "≈200 MB";
    this.pages = new Map();      // "fileId:page" -> [{lines:[lineIdx], vec}]
    this.worker = null; this.seq = 0; this.pending = new Map();
    this.db = null;
    this.queue = Promise.resolve();
  }

  emit() { this.dispatchEvent(new Event("change")); }

  /** Resume a previously enabled model (only called when the advanced option is on). */
  async init() {
    let on = false;
    try { on = localStorage.getItem("ls-semantic") === "1"; } catch { /* no storage */ }
    if (on) this.enable();
  }

  /** Stop the model; stored vectors stay so re-enabling is instant. */
  disable() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    for (const p of this.pending.values()) p.rej(new Error("semantic search turned off"));
    this.pending.clear();
    this.state = "off"; this.indexing = false;
    this.queue = Promise.resolve();
    try { localStorage.removeItem("ls-semantic"); } catch { /* no storage */ }
    this.emit();
  }

  call(cmd, payload = {}) {
    const id = ++this.seq;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.worker.postMessage({ cmd, id, ...payload });
    });
  }

  async enable() {
    if (this.state === "loading" || this.state === "ready") return;
    this.state = "loading"; this.error = ""; this.progress = 0; this.progressText = "";
    this.emit();
    try {
      if (!this.worker) {
        this.worker = new Worker(new URL("./embed.worker.js", import.meta.url), { type: "module" });
        this.worker.onmessage = (e) => {
          const d = e.data;
          if (d.type === "progress") { this.progress = d.frac; this.progressText = d.msg; this.emit(); return; }
          const p = this.pending.get(d.id);
          if (!p) return;
          this.pending.delete(d.id);
          d.type === "error" ? p.rej(new Error(d.error)) : p.res(d);
        };
      }
      this.db = this.db || await openDb();
      const rows = await req(this.db.transaction("pages").objectStore("pages").getAll());
      for (const r of rows) this.pages.set(r.fileId + ":" + r.page, r.chunks);
      const r = await this.call("init", { base: new URL("../", import.meta.url).href });
      this.eps = r.eps;
      this.state = "ready";
      try { localStorage.setItem("ls-semantic", "1"); } catch { /* no storage */ }
      this.emit();
      this.indexAll();
    } catch (e) {
      this.state = "error";
      this.error = String(e.message || e).slice(0, 200);
      this.emit();
    }
  }

  /** Embed every parsed page that has no vectors yet. */
  indexAll() {
    this.queue = this.queue.then(() => this._index()).catch((e) => {
      console.warn("indexing failed", e);
      this.indexing = false; this.emit();
    });
    return this.queue;
  }

  async _index() {
    if (this.state !== "ready" || !this.worker) return;
    const st = this.store;
    const todo = [];
    for (const [key, idxs] of st.pageLines) {
      const f = st.files.get(key.slice(0, key.lastIndexOf(":")));
      if (!f || f.status === "parsing" || this.pages.has(key) || !idxs.length) continue;
      todo.push(key);
    }
    if (!todo.length) return;
    this.indexing = true;
    const t0 = performance.now();
    const total = todo.length;
    let done = 0;
    // flatten into passages, batch across pages
    let buf = [];
    const flush = async () => {
      if (!buf.length) return;
      const { vecs } = await this.call("embed", { texts: buf.map((b) => b.text), kind: "doc" });
      const byPage = new Map();
      buf.forEach((b, i) => {
        if (!byPage.has(b.key)) byPage.set(b.key, []);
        byPage.get(b.key).push({ lines: b.lines, vec: vecs.slice(i * DIMS, (i + 1) * DIMS) });
      });
      const tx = this.db.transaction("pages", "readwrite").objectStore("pages");
      for (const [key, chunks] of byPage) {
        const prev = this.pages.get(key) || [];
        const all = prev.concat(chunks);
        this.pages.set(key, all);
        const i = key.lastIndexOf(":");
        tx.put({ fileId: key.slice(0, i), page: +key.slice(i + 1), chunks: all });
      }
      buf = [];
    };
    for (const key of todo) {
      const idxs = st.pageLines.get(key) || [];
      if (!st.files.has(key.slice(0, key.lastIndexOf(":")))) continue;   // deleted meanwhile
      const lines = idxs.map((d) => st.docs[d].text);
      for (const [ls, text] of passages(lines)) {
        buf.push({ key, lines: ls, text });
        if (buf.length >= BATCH) await flush();
      }
      done++;
      this.progress = done / total;
      const secs = (performance.now() - t0) / 1000;
      const eta = done > 8 ? ` · ~${Math.ceil((total - done) * secs / done)}s left` : "";
      this.progressText = `${done} / ${total} pages${eta} · ${this.eps.join("+")}`;
      if (done % 2 === 0) this.emit();
    }
    await flush();
    this.indexing = false;
    this.emit();
  }

  /** Top-k passages: [{docs:[docIdx], score}] best first. */
  async search(q, k = 20) {
    const { vecs: qv } = await this.call("embed", { texts: [q], kind: "query" });
    const st = this.store;
    const top = [];
    for (const [key, chunks] of this.pages) {
      const idxs = st.pageLines.get(key);
      if (!idxs) continue;
      for (const c of chunks) {
        let s = 0;
        for (let i = 0; i < DIMS; i++) s += qv[i] * c.vec[i];
        if (top.length < k || s > top[top.length - 1].score) {
          top.push({ score: s, docs: c.lines.map((l) => idxs[l]).filter((d) => d !== undefined) });
          top.sort((a, b) => b.score - a.score);
          if (top.length > k) top.pop();
        }
      }
    }
    return top;
  }

  onFileDone() { if (this.state === "ready") this.indexAll(); }

  async onFileDeleted(id) {
    for (const key of [...this.pages.keys()]) if (key.startsWith(id + ":")) this.pages.delete(key);
    if (!this.db) return;
    const s = this.db.transaction("pages", "readwrite").objectStore("pages");
    s.index("fileId").openCursor(IDBKeyRange.only(id)).onsuccess = (e) => {
      const c = e.target.result;
      if (c) { c.delete(); c.continue(); }
    };
  }

  clear() {
    this.pages.clear();
    if (this.db) this.db.transaction("pages", "readwrite").objectStore("pages").clear();
  }
}
