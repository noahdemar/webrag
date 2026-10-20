// Table-driven NFA interpreter — mirrors the WGSL kernel exactly.
// Docs: cells Uint32Array nDocs*CAP of code points + docLen Uint32Array.
// Returns per doc: [[start,end), ...] leftmost-longest non-overlapping matches.
import { isWord } from "./regex.js";

const CAP = 512;

function classesOf(tab, cp) {
  if (cp < 65536) return tab.cpBmp[cp];
  // astral: scan each class's astral ranges; bit 0 (".") always set
  let m = 1;
  for (let c = 1; c < tab.nClasses; c++) {
    const a = tab.clsRangeOff[c], b = tab.clsRangeOff[c + 1];
    for (let r = a; r < b; r += 2)
      if (cp >= tab.clsRanges[r] && cp <= tab.clsRanges[r + 1]) { m |= 1 << c; break; }
  }
  return m;
}

/** eps-expand: evaluate gates against (prevCp, curCp) to fixed point. */
function expand(tab, lo, hi, prevCp, curCp) {
  const { kind, fire, gateMask } = tab;
  for (let iter = 0; iter < tab.nStates; iter++) {
    let changed = false;
    let g = lo & gateMask[0], gh = hi & gateMask[1];
    while (g || gh) {
      const s = g ? 31 - Math.clz32(g) : 32 + (31 - Math.clz32(gh));
      const bitLo = s < 32 ? 1 << s : 0, bitHi = s >= 32 ? 1 << (s - 32) : 0;
      const wb = isWord(prevCp) !== isWord(curCp);
      const pass = kind[s] === 2 ? wb : !wb;
      // remove gate either way; on pass, fire its closure
      lo &= ~bitLo; hi &= ~bitHi;
      if (pass) { lo |= fire[s * 2]; hi |= fire[s * 2 + 1]; }
      changed = true;
      g &= ~bitLo; gh &= ~bitHi;
    }
    if (!changed) break;
  }
  return [lo, hi];
}

/**
 * Search one doc. cells slice + len -> array of [s,e) match spans.
 */
export function searchDoc(tab, cells, len) {
  const out = [];
  const { kind, cls, fire, consumeMask, acceptMask, start } = tab;
  let p = 0;
  while (p < len) {
    const prev = p > 0 ? cells[p - 1] : -1;
    let [lo, hi] = expand(tab, start[0], start[1], prev, cells[p]);
    let lastAcc = -1;
    let q = p;
    while (q < len) {
      const cp = cells[q];
      const cpCls = classesOf(tab, cp);
      let blo = 0, bhi = 0;
      let a = lo & consumeMask[0], ah = hi & consumeMask[1];
      while (a || ah) {
        const s = a ? 31 - Math.clz32(a) : 32 + (31 - Math.clz32(ah));
        if ((cpCls >> cls[s]) & 1) { blo |= fire[s * 2]; bhi |= fire[s * 2 + 1]; }
        if (s < 32) a &= ~(1 << s); else ah &= ~(1 << (s - 32));
      }
      if (!blo && !bhi) break;
      const nprev = cp;
      const ncur = q + 1 < len ? cells[q + 1] : -1;
      [lo, hi] = expand(tab, blo, bhi, nprev, ncur);
      q++;
      if ((lo & acceptMask[0]) | (hi & acceptMask[1])) lastAcc = q;
    }
    if (lastAcc >= 0) { out.push([p, lastAcc]); p = lastAcc; }
    else p++;
  }
  return out;
}

/**
 * Search all docs. cells Uint32Array (nDocs*CAP), docLen Uint32Array(nDocs).
 * maxMatchesPerDoc caps output size. Returns Array(nDocs) of span arrays.
 */
export function searchAll(tab, cells, docLen, maxMatchesPerDoc = 64, onDoc = null) {
  const n = docLen.length;
  const res = new Array(n);
  for (let d = 0; d < n; d++) {
    const len = docLen[d];
    if (!len) { res[d] = null; continue; }
    const spans = searchDoc(tab, cells.subarray(d * CAP, d * CAP + len), len);
    res[d] = spans.length ? spans.slice(0, maxMatchesPerDoc) : null;
    if (onDoc && (d & 0x3ff) === 0) onDoc(d, n);
  }
  return res;
}
