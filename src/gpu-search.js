// WebGPU search backend: table-driven NFA kernel, one thread per document.
// Mirrors src/search.js exactly: leftmost-longest non-overlapping, never empty.
import { FUZZY_WGSL } from "./fuzzy.js";

export const CAP = 512;
export const MAX_MATCHES = 32;

const WGSL = /* wgsl */ `
struct Params {
  nDocs: u32, cap: u32, nStates: u32, nClasses: u32,
  startLo: u32, startHi: u32,
  consumeLo: u32, consumeHi: u32,
  gateLo: u32, gateHi: u32,
  acceptLo: u32, acceptHi: u32,
  maxMatches: u32, pad0: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@group(0) @binding(2) var<storage, read> docLen: array<u32>;
@group(0) @binding(3) var<storage, read> kindCls: array<u32>;   // kind | cls<<8 per state
@group(0) @binding(4) var<storage, read> fire: array<u32>;
@group(0) @binding(5) var<storage, read> cpBmp: array<u32>;
@group(0) @binding(6) var<storage, read> clsRanges: array<u32>;
@group(0) @binding(7) var<storage, read> clsRangeOff: array<u32>;
@group(0) @binding(8) var<storage, read_write> outBuf: array<u32>;

fn kindOf(s: u32) -> u32 { return kindCls[s] & 0xffu; }
fn clsOf(s: u32) -> u32 { return (kindCls[s] >> 8u) & 0xffu; }

fn isWordf(cp: i32) -> bool {
  return (cp >= 48 && cp <= 57) || (cp >= 65 && cp <= 90) || cp == 95 || (cp >= 97 && cp <= 122);
}

fn cpClasses(cp: u32) -> u32 {
  if (cp < 65536u) { return cpBmp[cp]; }
  var m = 1u;
  for (var c = 1u; c < params.nClasses; c++) {
    let a = clsRangeOff[c];
    let b = clsRangeOff[c + 1u];
    for (var r = a; r < b; r += 2u) {
      if (cp >= clsRanges[r] && cp <= clsRanges[r + 1u]) { m |= 1u << c; break; }
    }
  }
  return m;
}

// eps-expand: evaluate gates against (prevCp, curCp) to fixed point
fn expand(loIn: u32, hiIn: u32, prevCp: i32, curCp: i32) -> vec2<u32> {
  var lo = loIn;
  var hi = hiIn;
  for (var iter = 0u; iter < params.nStates; iter++) {
    var changed = false;
    var g = lo & params.gateLo;
    var gh = hi & params.gateHi;
    while (g != 0u || gh != 0u) {
      var s: u32;
      var bitLo = 0u;
      var bitHi = 0u;
      if (g != 0u) { s = countTrailingZeros(g); bitLo = 1u << s; g &= g - 1u; }
      else { s = 32u + countTrailingZeros(gh); bitHi = 1u << (s - 32u); gh &= gh - 1u; }
      let wb = isWordf(prevCp) != isWordf(curCp);
      let k = kindOf(s);
      let gateOk = (k == 2u && wb) || (k == 3u && !wb);
      lo &= ~bitLo;
      hi &= ~bitHi;
      if (gateOk) { lo |= fire[s * 2u]; hi |= fire[s * 2u + 1u]; }
      changed = true;
    }
    if (!changed) { break; }
  }
  return vec2<u32>(lo, hi);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= params.nDocs) { return; }
  let outBase = d * (1u + 2u * params.maxMatches);
  let len = min(docLen[d], params.cap);
  if (len == 0u) { outBuf[outBase] = 0u; return; }
  let base = d * params.cap;
  var nOut = 0u;
  var p = 0u;
  while (p < len) {
    var prevCp = -1;
    if (p > 0u) { prevCp = i32(cells[base + p - 1u]); }
    var act = expand(params.startLo, params.startHi, prevCp, i32(cells[base + p]));
    var lastAcc = -1;
    var q = p;
    while (q < len) {
      let cp = cells[base + q];
      let cpm = cpClasses(cp);
      var blo = 0u;
      var bhi = 0u;
      var a = act.x & params.consumeLo;
      var ah = act.y & params.consumeHi;
      while (a != 0u || ah != 0u) {
        var s: u32;
        if (a != 0u) { s = countTrailingZeros(a); a &= a - 1u; }
        else { s = 32u + countTrailingZeros(ah); ah &= ah - 1u; }
        if (((cpm >> clsOf(s)) & 1u) != 0u) {
          blo |= fire[s * 2u];
          bhi |= fire[s * 2u + 1u];
        }
      }
      if (blo == 0u && bhi == 0u) { break; }
      var ncur = -1;
      if (q + 1u < len) { ncur = i32(cells[base + q + 1u]); }
      act = expand(blo, bhi, i32(cp), ncur);
      q += 1u;
      if ((act.x & params.acceptLo) != 0u || (act.y & params.acceptHi) != 0u) {
        lastAcc = i32(q);
      }
    }
    if (lastAcc >= 0) {
      if (nOut < params.maxMatches) {
        outBuf[outBase + 1u + 2u * nOut] = p;
        outBuf[outBase + 2u + 2u * nOut] = u32(lastAcc);
      }
      nOut += 1u;
      p = u32(lastAcc);
    } else {
      p += 1u;
    }
  }
  outBuf[outBase] = nOut;
}
`;

function packU8(u8) {
  const n = (u8.length + 3) >> 2;
  const w = new Uint32Array(n);
  for (let i = 0; i < u8.length; i++) w[i >> 2] |= u8[i] << ((i & 3) * 8);
  return w;
}

export class GpuSearch {
  static async create() {
    if (!navigator.gpu) return null;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      const g = new GpuSearch(device);
      // a WGSL error doesn't throw — it yields a pipeline that writes nothing
      const bad = async (m) => (await m.getCompilationInfo()).messages.some((x) => x.type === "error");
      if (await bad(g.module)) return null;
      if (await bad(g.fuzzyModule)) { console.warn("fuzzy kernel failed to compile; using JS"); g.fuzzyPipeline = null; }
      return g;
    } catch {
      return null;
    }
  }

  constructor(device) {
    this.device = device;
    this.module = device.createShaderModule({ code: WGSL });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: this.module, entryPoint: "main" },
    });
    this.fuzzyModule = device.createShaderModule({ code: FUZZY_WGSL });
    this.fuzzyPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: this.fuzzyModule, entryPoint: "main" },
    });
    this.fuzzyParams = new Uint32Array(8);
    this.fuzzyParamsBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fuzzyPatBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.cellCap = 0; this.docCap = 0;
    this.cellsBuf = null; this.docLenBuf = null; this.outBuf = null; this.readBuf = null;
    this.nDocs = 0;
    this.tableBufs = null;
    this.outStride = 1 + 2 * MAX_MATCHES;
  }

  mkBuf(arr, usage) {
    const b = this.device.createBuffer({
      size: Math.max(16, Math.ceil(arr.byteLength / 16) * 16),
      usage,
      mappedAtCreation: true,
    });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    b.unmap();
    return b;
  }

  /** Upload NFA tables for the current pattern. */
  setPattern(tab) {
    const u32 = (a) => (a instanceof Uint32Array ? a : Uint32Array.from(a));
    const kindClsArr = new Uint32Array(tab.nStates);
    for (let s = 0; s < tab.nStates; s++) kindClsArr[s] = tab.kind[s] | (tab.cls[s] << 8);
    const bufs = {
      kindCls: this.mkBuf(kindClsArr, GPUBufferUsage.STORAGE),
      fire: this.mkBuf(u32(tab.fire), GPUBufferUsage.STORAGE),
      cpBmp: this.mkBuf(u32(tab.cpBmp), GPUBufferUsage.STORAGE),
      clsRanges: this.mkBuf(u32(tab.clsRanges), GPUBufferUsage.STORAGE),
      clsRangeOff: this.mkBuf(u32(tab.clsRangeOff), GPUBufferUsage.STORAGE),
    };
    if (this.tableBufs) for (const b of Object.values(this.tableBufs)) b.destroy();
    this.tableBufs = bufs;
    this.params = new Uint32Array(16);
    this.params.set([
      0, CAP, tab.nStates, tab.nClasses,
      tab.start[0], tab.start[1],
      tab.consumeMask[0], tab.consumeMask[1],
      tab.gateMask[0], tab.gateMask[1],
      tab.acceptMask[0], tab.acceptMask[1],
      MAX_MATCHES, 0, 0, 0,
    ]);
    this.paramsBuf = this.mkBuf(this.params, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.tab = tab;
  }

  /** (Re)upload the document cells. Called when docs change. */
  setDocs(cells, docLen) {
    const n = docLen.length;
    this.nDocs = n;
    if (!n) return;
    const neededCells = n * CAP, neededOut = n * this.outStride;
    if (neededCells > this.cellCap) {
      if (this.cellsBuf) this.cellsBuf.destroy();
      const cap = Math.max(neededCells, this.cellCap * 2, 1 << 16);
      this.cellsBuf = this.device.createBuffer({
        size: cap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.cellCap = cap;
    }
    if (neededOut > this.docCap * this.outStride) {
      if (this.outBuf) this.outBuf.destroy();
      if (this.readBuf) this.readBuf.destroy();
      const cap = Math.max(n, this.docCap * 2, 1024);
      this.outBuf = this.device.createBuffer({
        size: cap * this.outStride * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.readBuf = this.device.createBuffer({
        size: cap * this.outStride * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.docLenBuf = this.device.createBuffer({
        size: cap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.docCap = cap;
    }
    this.device.queue.writeBuffer(this.cellsBuf, 0, cells.buffer, cells.byteOffset, n * CAP * 4);
    this.device.queue.writeBuffer(this.docLenBuf, 0, docLen.buffer, docLen.byteOffset, n * 4);
  }

  /** Run the regex kernel; returns Array(nDocs) of [s,e) spans (null when none). */
  async run() {
    const { device, nDocs } = this;
    if (!nDocs || !this.tab) return [];
    this.params[0] = nDocs;   // nDocs arrives via setDocs after setPattern
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);
    const bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.cellsBuf } },
        { binding: 2, resource: { buffer: this.docLenBuf } },
        { binding: 3, resource: { buffer: this.tableBufs.kindCls } },
        { binding: 4, resource: { buffer: this.tableBufs.fire } },
        { binding: 5, resource: { buffer: this.tableBufs.cpBmp } },
        { binding: 6, resource: { buffer: this.tableBufs.clsRanges } },
        { binding: 7, resource: { buffer: this.tableBufs.clsRangeOff } },
        { binding: 8, resource: { buffer: this.outBuf } },
      ],
    });
    return this.dispatch(this.pipeline, bind, false);
  }

  /** Run the fuzzy kernel for f = compileFuzzy(...); spans are [s,e,errors]. */
  async runFuzzy(f) {
    const { device, nDocs } = this;
    if (!this.fuzzyPipeline) throw new Error("fuzzy kernel unavailable");
    if (!nDocs) return [];
    this.fuzzyParams.set([nDocs, CAP, f.cps.length, f.k, MAX_MATCHES, 0, 0, 0]);
    device.queue.writeBuffer(this.fuzzyParamsBuf, 0, this.fuzzyParams);
    device.queue.writeBuffer(this.fuzzyPatBuf, 0, f.cps);
    const bind = device.createBindGroup({
      layout: this.fuzzyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.fuzzyParamsBuf } },
        { binding: 1, resource: { buffer: this.cellsBuf } },
        { binding: 2, resource: { buffer: this.docLenBuf } },
        { binding: 3, resource: { buffer: this.fuzzyPatBuf } },
        { binding: 4, resource: { buffer: this.outBuf } },
      ],
    });
    return this.dispatch(this.fuzzyPipeline, bind, true);
  }

  async dispatch(pipeline, bind, fuzzy) {
    const { device, nDocs } = this;
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(nDocs / 64));
    pass.end();
    const outBytes = nDocs * this.outStride * 4;
    enc.copyBufferToBuffer(this.outBuf, 0, this.readBuf, 0, outBytes);
    device.queue.submit([enc.finish()]);
    await this.readBuf.mapAsync(GPUMapMode.READ, 0, outBytes);
    const raw = new Uint32Array(this.readBuf.getMappedRange(0, outBytes));
    const res = new Array(nDocs);
    for (let d = 0; d < nDocs; d++) {
      const n = raw[d * this.outStride];
      if (!n) { res[d] = null; continue; }
      const spans = [];
      for (let k = 0; k < Math.min(n, MAX_MATCHES); k++) {
        const a = raw[d * this.outStride + 1 + 2 * k], e = raw[d * this.outStride + 2 + 2 * k];
        spans.push(fuzzy ? [a & 0xffff, e, a >>> 16] : [a, e]);
      }
      res[d] = spans;
    }
    this.readBuf.unmap();
    return res;
  }
}
