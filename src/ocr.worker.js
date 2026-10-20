// Background OCR worker: owns onnxruntime sessions and the ported indra pipeline.
// Protocol:
//   {cmd:'init', modelsBase}  -> {ok:true, eps:[...]}
//   {cmd:'ocr', gray:ArrayBuffer, w, h, turn} -> {lines, markdown, blocks, ...}
//   {cmd:'order', boxes:Float32Array(4n), texts:string[], w, h, contentIdx} -> {lines, markdown, blocks}
import * as ort from "../vendor/ort/ort.webgpu.min.mjs";
import { OcrEngine } from "./ocr.js";

let engine = null;
let initP = null;   // init promise — ocr/order calls await this
let progress = (msg) => self.postMessage({ type: "progress", msg });
let webgpuDead = false;

async function loadSession(file, eps, opts = {}) {
  return ort.InferenceSession.create(file, {
    executionProviders: eps,
    graphOptimizationLevel: "all",
    ...opts,
  });
}

async function init(modelsBase, opts = {}) {
  ort.env.wasm.wasmPaths = modelsBase + "vendor/ort/";
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
  // wasm by default — webgpu EP crashes (wasm OOB) on these models under ort-web 1.30;
  // opt back in with ?gpu once ort is upgraded.
  const eps = ["wasm"];
  try {
    if (opts.webgpu && navigator.gpu) eps.unshift("webgpu");
  } catch { /* no gpu */ }
  ort.env.webgpu = ort.env.webgpu || {};
  ort.env.logLevel = "warning";

  const meta = await (await fetch(modelsBase + "models/meta.json")).json();
  progress("loading ONNX sessions…");
  const mk = async (name) => {
    const want = webgpuDead ? ["wasm"] : eps;
    try {
      return await loadSession(modelsBase + "models/" + name, want);
    } catch (e) {
      if (want.length > 1) {
        webgpuDead = true;
        progress(`webgpu failed for ${name}, using wasm for all models (${String(e).slice(0, 80)})`);
        return loadSession(modelsBase + "models/" + name, ["wasm"]);
      }
      throw e;
    }
  };
  const [det, rec, order, orient, script, recCjk] = await Promise.all([
    mk("det.onnx"), mk("rec.onnx"), mk("order.onnx"),
    mk("orient.onnx"), mk("script.onnx"), mk("rec_cjk.onnx"),
  ]);
  engine = new OcrEngine(ort, { det, rec, order, orient, script, recCjk }, meta);
  return { eps: webgpuDead ? ["wasm"] : eps, vocab: engine.vocab.length };
}

self.onmessage = async (ev) => {
  const d = ev.data;
  try {
    if (d.cmd === "init") {
      initP = init(d.modelsBase, d);
      const r = await initP;
      self.postMessage({ type: "init", id: d.id, ...r });
      return;
    }
    // anything else waits for the engine
    if (!initP) throw new Error("engine not initialized");
    await initP;
    if (d.cmd === "ocr") {
      const gray = new Uint8Array(d.gray);
      const page = await engine.ocrPage(gray, d.w, d.h, d.turn !== false);
      self.postMessage({ type: "ocr", id: d.id, ...page });
    } else if (d.cmd === "order") {
      const r = await engine.orderAndLayout(
        Float32Array.from(d.boxes), d.texts, d.w, d.h,
        { contentOrder: d.contentOrder !== false, contentIdx: d.contentIdx || null },
      );
      self.postMessage({ type: "order", id: d.id, ...r });
    }
  } catch (e) {
    self.postMessage({ type: "error", id: d.id, cmd: d.cmd, error: String(e && e.stack || e) });
  }
};
