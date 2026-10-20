// Document store: extracted lines -> fixed-size code-point cells (handwired
// json-search style), backed by IndexedDB for persistence + JSONL dumps.
//
// Doc = one text line. cells[d*CAP .. d*CAP+CAP) = code points; docLen[d] =
// real length (capped at CAP for the kernel; `fullLen` keeps the truth).
import { CAP } from "./gpu-search.js";

const DB_NAME = "localscrape", DB_VER = 1;

export function openDb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB_NAME, DB_VER);
    r.onupgradeneeded = () => {
      const db = r.result;
      db.createObjectStore("files", { keyPath: "id" });
      const pages = db.createObjectStore("pages", { keyPath: ["fileId", "page"] });
      pages.createIndex("fileId", "fileId");
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

// Original files live in the Origin Private File System so page previews and
// highlights keep working after a reload.
async function originalsDir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("originals", { create: true });
}

const tx = (db, store, mode) => db.transaction(store, mode).objectStore(store);
const req = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

function textToCells(text) {
  const cps = new Uint32Array(CAP);
  let len = 0, full = 0;
  for (const ch of text) {
    full++;
    if (len < CAP) cps[len++] = ch.codePointAt(0);
  }
  return { cps, len, full };
}

export class Store extends EventTarget {
  constructor() {
    super();
    this.docs = [];            // {fileId, fileName, page, lineIdx, box, text, fullLen}
    this.docLen = new Uint32Array(0);
    this.cells = new Uint32Array(0);
    this.cellUsed = 0;
    this.files = new Map();    // id -> {id,name,size,pageCount,lines,mode,status,addedAt,note}
    this.pageLines = new Map(); // `${fileId}:${page}` -> [docIdx]
    this.db = null;
    this.dirty = false;        // cells changed since last upload
  }

  async init() {
    this.db = await openDb();
    const files = await req(tx(this.db, "files", "readonly").getAll());
    for (const f of files) this.files.set(f.id, f);
    const pages = await req(tx(this.db, "pages", "readonly").getAll());
    pages.sort((a, b) => a.fileId.localeCompare(b.fileId) || a.page - b.page);
    for (const p of pages) this.addPage(p.fileId, p.page, p.lines, { persist: false });
  }

  async saveOriginal(id, file) {
    try {
      const fh = await (await originalsDir()).getFileHandle(id, { create: true });
      const w = await fh.createWritable();
      await w.write(file);
      await w.close();
      return true;
    } catch (e) {
      console.warn("could not keep original file", e);
      return false;
    }
  }

  async getOriginal(id) {
    try {
      return await (await (await originalsDir()).getFileHandle(id)).getFile();
    } catch {
      return null;
    }
  }

  async dropOriginals(id = null) {
    try {
      const root = await navigator.storage.getDirectory();
      if (id) await (await originalsDir()).removeEntry(id);
      else await root.removeEntry("originals", { recursive: true });
    } catch { /* nothing stored */ }
  }

  emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }

  /** Register a file record (persisted). */
  async addFile(f) {
    this.files.set(f.id, f);
    if (this.db) tx(this.db, "files", "readwrite").put(f);
    this.emit("files");
  }

  updateFile(id, patch) {
    const f = this.files.get(id);
    if (!f) return;
    Object.assign(f, patch);
    if (this.db) tx(this.db, "files", "readwrite").put(f);
    this.emit("files");
  }

  /** Append one extracted page's lines as docs. lines: [{box,text,conf}] */
  addPage(fileId, page, lines, { persist = true } = {}) {
    const f = this.files.get(fileId);
    const fileName = f ? f.name : "?";
    const key = `${fileId}:${page}`;
    let idxs = this.pageLines.get(key);
    if (!idxs) { idxs = []; this.pageLines.set(key, idxs); }
    for (const ln of lines) {
      if (!ln.text || !ln.text.trim()) continue;
      const { cps, len, full } = textToCells(ln.text);
      const d = this.docs.length;
      this.docs.push({
        fileId, fileName, page,
        lineIdx: idxs.length, box: ln.box, text: ln.text, fullLen: full,
      });
      idxs.push(d);
      if (this.cellUsed + CAP > this.cells.length) {
        const ncap = Math.max(this.cells.length * 2, 1 << 16, this.cellUsed + CAP);
        const nc = new Uint32Array(ncap);
        nc.set(this.cells.subarray(0, this.cellUsed));
        this.cells = nc;
      }
      this.cells.set(cps, this.cellUsed);
      this.cellUsed += CAP;
      if (this.docLen.length <= d) {
        const nd = new Uint32Array(Math.max(this.docLen.length * 2, 1024, d + 1));
        nd.set(this.docLen);
        this.docLen = nd;
      }
      this.docLen[d] = len;
    }
    this.dirty = true;
    if (persist && this.db) {
      tx(this.db, "pages", "readwrite").put({ fileId, page, lines });
    }
    this.emit("docs", { fileId, page, count: lines.length });
  }

  /** Lines of a page in reading order (doc indices). */
  pageDocIdxs(fileId, page) { return this.pageLines.get(`${fileId}:${page}`) || []; }

  /** Neighbouring docs of doc d on its page (for context lines). */
  context(d, n = 1) {
    const doc = this.docs[d];
    const idxs = this.pageDocIdxs(doc.fileId, doc.page);
    const pos = idxs.indexOf(d);
    const grab = (k) => (pos + k >= 0 && pos + k < idxs.length) ? this.docs[idxs[pos + k]] : null;
    return { before: [...Array(n)].map((_, k) => grab(k - n)).reverse().filter(Boolean), after: [...Array(n)].map((_, k) => grab(k + 1)).filter(Boolean) };
  }

  async deleteFile(id) {
    // drop docs belonging to the file (rebuild cells)
    const keep = [];
    const newDocLen = new Uint32Array(this.docs.length);
    const newCells = new Uint32Array(this.cells.length);
    let used = 0;
    this.pageLines = new Map();
    this.docs.forEach((doc, d) => {
      if (doc.fileId === id) return;
      const nd = keep.length;
      keep.push(doc);
      newDocLen[nd] = this.docLen[d];
      newCells.set(this.cells.subarray(d * CAP, d * CAP + CAP), used);
      used += CAP;
      const key = `${doc.fileId}:${doc.page}`;
      let idxs = this.pageLines.get(key);
      if (!idxs) { idxs = []; this.pageLines.set(key, idxs); }
      idxs.push(nd);
    });
    this.docs = keep;
    this.docLen = newDocLen;
    this.cells = newCells;
    this.cellUsed = used;
    this.files.delete(id);
    this.dirty = true;
    this.dropOriginals(id);
    if (this.db) {
      const t = this.db.transaction(["files", "pages"], "readwrite");
      t.objectStore("files").delete(id);
      t.objectStore("pages").index("fileId").openCursor(IDBKeyRange.only(id)).onsuccess = (e) => {
        const c = e.target.result;
        if (c) { c.delete(); c.continue(); }
      };
    }
    this.emit("files"); this.emit("docs");
  }

  async clearAll() {
    this.docs = []; this.docLen = new Uint32Array(0);
    this.cells = new Uint32Array(0); this.cellUsed = 0;
    this.files.clear(); this.pageLines.clear();
    this.dirty = true;
    this.dropOriginals();
    if (this.db) {
      const t = this.db.transaction(["files", "pages"], "readwrite");
      t.objectStore("files").clear(); t.objectStore("pages").clear();
    }
    this.emit("files"); this.emit("docs");
  }

  stats() {
    return {
      files: this.files.size,
      pages: this.pageLines.size,
      docs: this.docs.length,
      chars: [...this.docs].reduce((s, d) => s + d.fullLen, 0),
    };
  }

  // ---------- dumps ----------

  async exportDump(onProgress) {
    const files = await req(tx(this.db, "files", "readonly").getAll());
    const pages = await req(tx(this.db, "pages", "readonly").getAll());
    pages.sort((a, b) => a.fileId.localeCompare(b.fileId) || a.page - b.page);
    const byId = new Map(files.map((f) => [f.id, f]));
    const lines = [];
    for (const f of files) {
      lines.push(JSON.stringify({ v: 1, kind: "file", name: f.name, size: f.size, pageCount: f.pageCount, mode: f.mode, status: f.status, addedAt: f.addedAt }));
      for (const p of pages) {
        if (p.fileId !== f.id) continue;
        lines.push(JSON.stringify({ v: 1, kind: "page", file: f.name, page: p.page, lines: p.lines }));
        if (onProgress) onProgress(lines.length);
      }
    }
    return lines.join("\n") + "\n";
  }

  /** Import a dump (.jsonl). Groups pages by file name. Returns file count. */
  async importDump(text, onProgress) {
    const rows = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const fileRows = rows.filter((r) => r.kind === "file");
    const pageRows = rows.filter((r) => r.kind === "page");
    const byName = new Map();
    for (const f of fileRows) byName.set(f.name, f);
    // also synthesize file rows for pages lacking a header
    for (const p of pageRows) if (!byName.has(p.file)) byName.set(p.file, { name: p.file });
    let n = 0;
    for (const [name, fr] of byName) {
      const existing = [...this.files.values()].find((f) => f.name === name && f.size === fr.size);
      if (existing) await this.deleteFile(existing.id);
      const id = existing ? existing.id : crypto.randomUUID();
      const f = {
        id, name, size: fr.size || 0, addedAt: fr.addedAt || Date.now(),
        pageCount: fr.pageCount || 0, mode: fr.mode || "dump", status: "done", lines: 0, note: existing ? "re-imported" : "imported",
      };
      await this.addFile(f);
      const myPages = pageRows.filter((p) => p.file === name).sort((a, b) => a.page - b.page);
      let nLines = 0;
      for (const p of myPages) {
        this.addPage(id, p.page, p.lines || []);
        nLines += (p.lines || []).length;
      }
      this.updateFile(id, { lines: nLines, pageCount: f.pageCount || myPages.length });
      n++;
      if (onProgress) onProgress(n, byName.size);
    }
    return n;
  }
}
