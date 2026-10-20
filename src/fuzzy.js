// Fuzzy (approximate) search: Wu–Manber bitap with up to k edits, one thread
// per line on WebGPU, plus an identical JS fallback.
//
// OCR-tolerant normalization: case-folded, and spaces / hyphens are ignored in
// both pattern and text, so "important" finds "im portant" and "high-modernist"
// finds "high modernist". Ends come from bitap; starts from a short anchored
// backward edit-distance pass. Spans are [s,e) cell indices, like regex search.
export const FUZZY_MAX = 32;   // pattern letters (one u32 bitmask)

/** Case-fold one code point (ASCII, Latin-1, curly quotes). */
export function fold(c) {
  if (c >= 65 && c <= 90) return c + 32;
  if (c >= 0xc0 && c <= 0xde && c !== 0xd7) return c + 32;
  if (c === 0x2018 || c === 0x2019) return 39;
  if (c === 0x201c || c === 0x201d) return 34;
  return c;
}

/** Characters the matcher steps over (whitespace, hyphens, soft hyphen). */
export function isSkip(c) {
  return c === 32 || c === 9 || c === 10 || c === 13 || c === 0xa0 || c === 0xad ||
    c === 45 || (c >= 0x2010 && c <= 0x2015);
}

/** Query -> {cps: Uint32Array, k, error}. */
export function compileFuzzy(src) {
  const cps = [];
  for (const ch of src) {
    const c = ch.codePointAt(0);
    if (!isSkip(c)) cps.push(fold(c));
  }
  if (!cps.length) return { error: "type something to search" };
  if (cps.length > FUZZY_MAX) return { error: `fuzzy search takes up to ${FUZZY_MAX} letters` };
  const m = cps.length;
  const k = m <= 4 ? 0 : m <= 8 ? 1 : m <= 14 ? 2 : 3;
  return { cps: Uint32Array.from(cps), k };
}

// ---------- JS reference (mirrors the WGSL kernel) ----------

function startOf(cells, e, pat, m, err) {
  // anchored at e, walk back until reversed pattern is matched within `err`
  let D = new Uint32Array(m + 1), N = new Uint32Array(m + 1);
  for (let i = 0; i <= m; i++) D[i] = i;
  // keep the lowest-cost start (ties -> farther back); stop once no column can improve
  const lim = Math.max(0, e - 4 * (m + err) - 8);
  let best = e - 1, bestD = 99;
  for (let t = e - 1; t >= lim; t--) {
    const c = cells[t];
    if (isSkip(c)) continue;
    const fc = fold(c);
    N[0] = D[0] + 1;
    let colMin = N[0];
    for (let i = 1; i <= m; i++) {
      const sub = D[i - 1] + (pat[m - i] === fc ? 0 : 1);
      N[i] = Math.min(D[i] + 1, N[i - 1] + 1, sub);
      colMin = Math.min(colMin, N[i]);
    }
    [D, N] = [N, D];
    if (D[m] <= bestD) { bestD = D[m]; best = t; }
    if (colMin > bestD) break;
  }
  return best;
}

export function fuzzyDoc(pat, k, cells, len, maxOut = 32) {
  const m = pat.length, hit = 1 << (m - 1);
  const R = new Int32Array(k + 1);
  const reset = () => { for (let j = 0; j <= k; j++) R[j] = (1 << j) - 1; };
  reset();
  const out = [];
  let candE = -1, candErr = 99, prevQ = -2;
  const emit = () => {
    out.push([startOf(cells, candE, pat, m, candErr), candE, candErr]);
    candE = -1; candErr = 99; reset();
  };
  for (let q = 0; q < len && out.length < maxOut; q++) {
    const c = cells[q];
    if (isSkip(c)) { prevQ = -2; continue; }   // a gap ends tie-extension
    const fc = fold(c);
    let mask = 0;
    for (let i = 0; i < m; i++) if (pat[i] === fc) mask |= 1 << i;
    let oldPrev = R[0];
    R[0] = ((R[0] << 1) | 1) & mask;
    for (let j = 1; j <= k; j++) {
      const old = R[j];
      R[j] = (((old << 1) | 1) & mask) | oldPrev | ((oldPrev << 1) | 1) | ((R[j - 1] << 1) | 1);
      oldPrev = old;
    }
    let bestErr = 99;
    for (let j = 0; j <= k; j++) if (R[j] & hit) { bestErr = j; break; }
    if (bestErr <= k) {
      // lower cost wins; an equal cost on the very next char extends the match
      if (bestErr < candErr || (bestErr === candErr && prevQ === candE - 1)) { candErr = bestErr; candE = q + 1; }
    } else if (candE >= 0) emit();
    prevQ = q;
  }
  if (candE >= 0 && out.length < maxOut) emit();
  return out;
}

export function fuzzyAll(f, cells, docLen, CAP) {
  const n = docLen.length, res = new Array(n);
  for (let d = 0; d < n; d++) {
    const len = docLen[d];
    if (!len) { res[d] = null; continue; }
    const spans = fuzzyDoc(f.cps, f.k, cells.subarray(d * CAP, d * CAP + len), len);
    res[d] = spans.length ? spans : null;
  }
  return res;
}

// ---------- WGSL ----------

export const FUZZY_WGSL = /* wgsl */ `
struct FParams { nDocs: u32, cap: u32, m: u32, k: u32, maxMatches: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> fp: FParams;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@group(0) @binding(2) var<storage, read> docLen: array<u32>;
@group(0) @binding(3) var<storage, read> pat: array<u32>;
@group(0) @binding(4) var<storage, read_write> outBuf: array<u32>;

fn fold(c: u32) -> u32 {
  if (c >= 65u && c <= 90u) { return c + 32u; }
  if (c >= 0xc0u && c <= 0xdeu && c != 0xd7u) { return c + 32u; }
  if (c == 0x2018u || c == 0x2019u) { return 39u; }
  if (c == 0x201cu || c == 0x201du) { return 34u; }
  return c;
}
fn isSkip(c: u32) -> bool {
  return c == 32u || c == 9u || c == 10u || c == 13u || c == 0xa0u || c == 0xadu ||
    c == 45u || (c >= 0x2010u && c <= 0x2015u);
}

fn startOf(base: u32, e: u32, err: u32) -> u32 {
  let m = fp.m;
  var D: array<u32, 33>;
  var N: array<u32, 33>;
  for (var i = 0u; i <= m; i++) { D[i] = i; }
  let span = 4u * (m + err) + 8u;
  var lim = 0u;
  if (e > span) { lim = e - span; }
  var best = e - 1u;
  var bestD = 99u;
  var t = e;
  while (t > lim) {
    t -= 1u;
    let c = cells[base + t];
    if (isSkip(c)) { continue; }
    let fc = fold(c);
    N[0] = D[0] + 1u;
    var colMin = N[0];
    for (var i = 1u; i <= m; i++) {
      var sub = D[i - 1u];
      if (pat[m - i] != fc) { sub += 1u; }
      N[i] = min(min(D[i] + 1u, N[i - 1u] + 1u), sub);
      colMin = min(colMin, N[i]);
    }
    for (var i = 0u; i <= m; i++) { D[i] = N[i]; }
    if (D[m] <= bestD) { bestD = D[m]; best = t; }
    if (colMin > bestD) { break; }
  }
  return best;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= fp.nDocs) { return; }
  let outBase = d * (1u + 2u * fp.maxMatches);
  let len = min(docLen[d], fp.cap);
  let base = d * fp.cap;
  let m = fp.m;
  let k = fp.k;
  let hit = 1u << (m - 1u);
  var R: array<u32, 8>;
  for (var j = 0u; j <= k; j++) { R[j] = (1u << j) - 1u; }
  var nOut = 0u;
  var candE = 0u;
  var candErr = 99u;
  var prevQ = 0xffffffffu;
  for (var q = 0u; q < len; q++) {
    if (nOut >= fp.maxMatches) { break; }
    let c = cells[base + q];
    if (isSkip(c)) { prevQ = 0xfffffff0u; continue; }
    let fc = fold(c);
    var mask = 0u;
    for (var i = 0u; i < m; i++) { if (pat[i] == fc) { mask |= 1u << i; } }
    var oldPrev = R[0];
    R[0] = ((R[0] << 1u) | 1u) & mask;
    for (var j = 1u; j <= k; j++) {
      let old = R[j];
      R[j] = (((old << 1u) | 1u) & mask) | oldPrev | ((oldPrev << 1u) | 1u) | ((R[j - 1u] << 1u) | 1u);
      oldPrev = old;
    }
    var bestErr = 99u;
    for (var j = 0u; j <= k; j++) { if ((R[j] & hit) != 0u) { bestErr = j; break; } }
    let grow = candErr <= k && prevQ + 1u == candE;
    prevQ = q;
    if (bestErr <= k) {
      if (bestErr < candErr || (bestErr == candErr && grow)) { candErr = bestErr; candE = q + 1u; }
    } else if (candErr <= k) {
      outBuf[outBase + 1u + 2u * nOut] = startOf(base, candE, candErr) | (candErr << 16u);
      outBuf[outBase + 2u + 2u * nOut] = candE;
      nOut += 1u;
      candErr = 99u;
      for (var j = 0u; j <= k; j++) { R[j] = (1u << j) - 1u; }
    }
  }
  if (candErr <= k && nOut < fp.maxMatches) {
    outBuf[outBase + 1u + 2u * nOut] = startOf(base, candE, candErr) | (candErr << 16u);
    outBuf[outBase + 2u + 2u * nOut] = candE;
    nOut += 1u;
  }
  outBuf[outBase] = nOut;
}
`;
