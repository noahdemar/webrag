// Embedding worker: EmbeddingGemma-300M (q4 ONNX) via onnxruntime-web.
// The model is fetched once from Hugging Face and kept in Cache Storage; text
// never leaves the browser. Protocol:
//   {cmd:'init', base}                     -> {eps, dims}
//   {cmd:'embed', texts:[...], kind:'doc'|'query'} -> {vecs: Float32Array(n*DIMS)}
import * as ort from "../vendor/ort/ort.webgpu.min.mjs";
import { Tokenizer } from "../vendor/tokenizers.min.mjs";

const HF = "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/";
const FILES = {
  model: "onnx/model_q4.onnx",
  data: "onnx/model_q4.onnx_data",
  tok: "tokenizer.json",
  tokCfg: "tokenizer_config.json",
};
export const DIMS = 256;     // Matryoshka truncation (768 -> 256)
const MAX_TOKENS = 256;
const CACHE = "localscrape-models-v1";

let session = null, tokenizer = null, initP = null;
const progress = (msg, frac) => self.postMessage({ type: "progress", msg, frac });

/** fetch with Cache Storage + byte progress across all files. */
async function fetchCached(names) {
  const cache = await caches.open(CACHE);
  const hits = await Promise.all(names.map((n) => cache.match(HF + n)));
  const out = new Array(names.length);
  let total = 0, got = 0;
  const heads = await Promise.all(names.map(async (n, i) => {
    if (hits[i]) return 0;
    const r = await fetch(HF + n, { method: "HEAD" });
    return +r.headers.get("content-length") || 0;
  }));
  total = heads.reduce((a, b) => a + b, 0);
  for (let i = 0; i < names.length; i++) {
    if (hits[i]) { out[i] = await hits[i].arrayBuffer(); continue; }
    const r = await fetch(HF + names[i]);
    if (!r.ok) throw new Error(`download failed: ${names[i]} (${r.status})`);
    const reader = r.body.getReader();
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      if (total) progress(`Downloading model… ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, got / total);
    }
    const blob = new Blob(chunks);
    await cache.put(HF + names[i], new Response(blob));
    out[i] = await blob.arrayBuffer();
  }
  return out;
}

async function init(base) {
  ort.env.wasm.wasmPaths = base + "vendor/ort/";
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
  ort.env.logLevel = "error";
  progress("Checking model cache…", 0);
  const [model, data, tok, tokCfg] = await fetchCached([FILES.model, FILES.data, FILES.tok, FILES.tokCfg]);
  const dec = new TextDecoder();
  tokenizer = new Tokenizer(JSON.parse(dec.decode(tok)), JSON.parse(dec.decode(tokCfg)));
  progress("Starting model…", 1);
  const opts = (eps) => ({
    executionProviders: eps,
    graphOptimizationLevel: "all",
    externalData: [{ path: "model_q4.onnx_data", data: new Uint8Array(data) }],
  });
  const tries = navigator.gpu ? [["webgpu"], ["wasm"]] : [["wasm"]];
  let err = null;
  for (const eps of tries) {
    try {
      session = await ort.InferenceSession.create(new Uint8Array(model), opts(eps));
      // smoke test: webgpu can fail at first run rather than at create
      await embed(["warm up"], "query");
      return { eps, dims: DIMS };
    } catch (e) { err = e; session = null; }
  }
  throw err;
}

const PREFIX = { query: "task: search result | query: ", doc: "title: none | text: " };

async function embed(texts, kind) {
  const enc = texts.map((t) => tokenizer.encode(PREFIX[kind] + t).ids.slice(0, MAX_TOKENS));
  const L = Math.max(...enc.map((e) => e.length));
  const n = enc.length;
  const ids = new BigInt64Array(n * L), mask = new BigInt64Array(n * L);
  enc.forEach((e, i) => e.forEach((id, j) => { ids[i * L + j] = BigInt(id); mask[i * L + j] = 1n; }));
  const out = await session.run({
    input_ids: new ort.Tensor("int64", ids, [n, L]),
    attention_mask: new ort.Tensor("int64", mask, [n, L]),
  });
  const emb = out.sentence_embedding;
  const D = emb.dims[1], src = emb.data;
  const vecs = new Float32Array(n * DIMS);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < DIMS; k++) { const v = src[i * D + k]; vecs[i * DIMS + k] = v; s += v * v; }
    s = 1 / Math.sqrt(s || 1);
    for (let k = 0; k < DIMS; k++) vecs[i * DIMS + k] *= s;
  }
  return vecs;
}

self.onmessage = async (ev) => {
  const d = ev.data;
  try {
    if (d.cmd === "init") {
      initP = initP || init(d.base);
      self.postMessage({ type: "init", id: d.id, ...(await initP) });
    } else if (d.cmd === "embed") {
      await initP;
      const vecs = await embed(d.texts, d.kind);
      self.postMessage({ type: "embed", id: d.id, vecs }, [vecs.buffer]);
    }
  } catch (e) {
    if (d.cmd === "init") initP = null;
    self.postMessage({ type: "error", id: d.id, error: String(e && e.message || e) });
  }
};
