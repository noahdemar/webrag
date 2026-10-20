// Regex subset -> table-driven NFA (handwired json-search style).
//
// Supported: literals, '.', [cls] with ranges and ^negation, \d \D \w \W \s \S
// \n \t \r \b \B + escaped metas, ( ) groups, | alternation, ? * + {n} {n,} {n,m}.
// Match semantics: leftmost-longest, non-overlapping, never empty.
//
// Compiled tables (all u32):
//   kind:     Uint8Array  0=CONSUME 1=EPS 2=GATE_WB 3=GATE_NWB 4=ACCEPT
//   cls:      Uint8Array  char-class id per CONSUME state (0 = any)
//   fire:     Uint32Array 2*S — for CONSUME/GATE s, eps0-closure of its target
//   eps0:     Uint32Array 2*S — eps-closure of each state (gates are barriers)
//   start:    u64 mask [2] — eps0[start]
//   masks:    consumeMask, gateMask, acceptMask (u64 as [lo,hi])
//   cpBmp:    Uint32Array(65536) — bit c set if cp in class c (BMP fast path)
//   clsRanges:Uint32Array — [lo,hi] pairs per class for astral cps
//   clsRangeOff: Uint32Array — start index into clsRanges per class (+1)
//   nStates, nClasses, hasGates, ignoreCase

export class PatternError extends Error {}

const KIND = { CONSUME: 0, EPS: 1, GATE_WB: 2, GATE_NWB: 3, ACCEPT: 4 };
const MAX_STATES = 64, MAX_CLASSES = 32;
const ANY = 0; // class id reserved for "."
const MAX_CP = 0x10ffff;

export const isWord = (cp) =>
  (cp >= 48 && cp <= 57) || (cp >= 65 && cp <= 90) || cp === 95 || (cp >= 97 && cp <= 122);

// ---------- char classes -------------------------------------------------

const CLASS_ESCAPES = {
  d: [[48, 57]],
  w: [[48, 57], [65, 90], [95], [97, 122]],
  s: [[9, 13], [32, 32], [0xa0, 0xa0], [0x1680, 0x1680], [0x2000, 0x200a], [0x2028, 0x2029], [0x202f, 0x202f], [0x205f, 0x205f], [0x3000, 0x3000], [0xfeff, 0xfeff]],
};

function foldRanges(ranges, icase) {
  if (!icase) return ranges;
  const extra = [];
  for (const [lo, hi] of ranges) {
    for (let cp = lo; cp <= Math.min(hi, 0xffff); cp++) {
      const c = String.fromCodePoint(cp);
      for (const v of [c.toUpperCase(), c.toLowerCase()]) {
        if (v.length === 1) {
          const vpc = v.codePointAt(0);
          if (vpc !== cp) extra.push([vpc, vpc]);
        }
      }
    }
  }
  return ranges.concat(extra);
}

function normRanges(ranges) {
  const s = ranges.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  for (const r of s) {
    if (r[0] > r[1]) continue;
    const last = out[out.length - 1];
    if (last && r[0] <= last[1] + 1) last[1] = Math.max(last[1], r[1]);
    else out.push([r[0], r[1]]);
  }
  return out;
}

function complement(ranges) {
  const n = normRanges(ranges);
  const out = [];
  let lo = 0;
  for (const [a, b] of n) {
    if (a > lo) out.push([lo, a - 1]);
    lo = Math.max(lo, b + 1);
  }
  if (lo <= MAX_CP) out.push([lo, MAX_CP]);
  return out;
}

// ---------- parser -------------------------------------------------------

const META = "\\^$.*+?()[]{}|";

function parse(src) {
  let i = 0;
  const peek = () => src[i];
  const err = (m) => new PatternError(`${m} at offset ${i} in /${src}/`);

  function parseEscape(inClass) {
    i++; // skip backslash
    if (i >= src.length) throw err("trailing backslash");
    const c = src[i++];
    if (c in CLASS_ESCAPES) return { cls: CLASS_ESCAPES[c] };
    if (c === "D" || c === "W" || c === "S")
      return { cls: complement(CLASS_ESCAPES[c.toLowerCase()]) };
    const simple = { n: 10, t: 9, r: 13, v: 11, f: 12, 0: 0, a: 7, e: 27 };
    if (c in simple) return { cp: simple[c] };
    if (c === "b" && !inClass) return { gate: "wb" };
    if (c === "B" && !inClass) return { gate: "nwb" };
    if (c === "x") {
      const h = src.slice(i, i + 2);
      if (!/^[0-9a-fA-F]{2}$/.test(h)) throw err("bad \\x escape");
      i += 2;
      return { cp: parseInt(h, 16) };
    }
    if (c === "u") {
      let h;
      if (src[i] === "{") {
        const end = src.indexOf("}", i);
        if (end < 0) throw err("unclosed \\u{");
        h = src.slice(i + 1, end);
        i = end + 1;
      } else {
        h = src.slice(i, i + 4);
        i += 4;
      }
      if (!/^[0-9a-fA-F]{1,6}$/.test(h)) throw err("bad \\u escape");
      const cp = parseInt(h, 16);
      if (cp > MAX_CP) throw err("\\u escape out of range");
      return { cp };
    }
    if (META.includes(c) || !/[a-zA-Z]/.test(c)) return { cp: c.codePointAt(0) };
    throw err(`unsupported escape \\${c}`);
  }

  function parseClass() {
    i++; // [
    let neg = false;
    if (peek() === "^") { neg = true; i++; }
    const ranges = [];
    let first = true;
    while (true) {
      if (i >= src.length) throw err("unclosed [");
      if (src[i] === "]" && !first) { i++; break; }
      first = false;
      let lo;
      if (src[i] === "\\") {
        const e = parseEscape(true);
        if (e.cls) { ranges.push(...e.cls); continue; }
        lo = e.cp;
      } else {
        lo = src.codePointAt(i);
        i += src[i] <= "\uffff" ? 1 : 2;
      }
      // range?
      if (src[i] === "-" && src[i + 1] !== "]") {
        i++;
        let hi;
        if (src[i] === "\\") {
          const e = parseEscape(true);
          if (e.cls) throw err("class escape on range right side");
          hi = e.cp;
        } else {
          hi = src.codePointAt(i);
          i += src[i] <= "\uffff" ? 1 : 2;
        }
        if (hi < lo) throw err("range out of order");
        ranges.push([lo, hi]);
      } else {
        ranges.push([lo, lo]);
      }
    }
    return { cls: neg ? complement(ranges) : ranges };
  }

  function parseAtom() {
    const c = peek();
    if (c === "(") {
      i++;
      if (src.startsWith("?", i)) {
        // only non-capturing (?:...) accepted; others are unsupported
        if (!src.startsWith("?:", i)) throw err("only (?:...) groups supported");
        i += 3;
      }
      const a = parseAlt();
      if (peek() !== ")") throw err("unclosed (");
      i++;
      return a;
    }
    if (c === "[") { const { cls } = parseClass(); return { type: "char", cls }; }
    if (c === ".") { i++; return { type: "char", cls: null }; }
    if (c === "\\") {
      const e = parseEscape(false);
      if (e.gate) return { type: "gate", gate: e.gate };
      if (e.cls) return { type: "char", cls: e.cls };
      return { type: "char", cls: [[e.cp, e.cp]] };
    }
    if ("|)".includes(c)) throw err("empty alternative");
    if ("*+?".includes(c)) throw err(`nothing to repeat '${c}'`);
    const cp = src.codePointAt(i);
    i += src[i] <= "\uffff" ? 1 : 2;
    return { type: "char", cls: [[cp, cp]] };
  }

  function parseRep() {
    let a = parseAtom();
    while (i < src.length) {
      const c = peek();
      let q = null;
      if (c === "?" || c === "*" || c === "+") { q = c; i++; }
      else if (c === "{") {
        const m = /^\{(\d+)(?:,(\d*)?)?\}/.exec(src.slice(i, i + 12));
        if (!m) { break; }
        i += m[0].length;
        const n0 = parseInt(m[1], 10);
        if (m[1] !== undefined && m[2] === undefined && m[0].includes(",")) q = [n0, Infinity];
        else if (m[0].includes(",")) q = [n0, m[2] === "" ? Infinity : parseInt(m[2], 10)];
        else q = [n0, n0];
      } else break;
      if (peek() === "?") i++; // lazy quantifier: same leftmost-longest semantics
      a = { type: "rep", q, a };
    }
    return a;
  }

  function parseConcat() {
    const parts = [];
    while (i < src.length && !"|)".includes(peek())) parts.push(parseRep());
    return { type: "cat", parts };
  }

  function parseAlt() {
    const alts = [parseConcat()];
    while (peek() === "|") { i++; alts.push(parseConcat()); }
    return { type: "alt", alts };
  }

  const ast = parseAlt();
  if (i !== src.length) throw err(`unexpected '${src[i]}'`);
  return ast;
}

// ---------- NFA build ----------------------------------------------------

function build(ast, icase) {
  // states: {kind, cls (ranges|null=any), eps: [targets]}
  const states = [];
  const newState = (kind, cls = null) => {
    if (states.length >= MAX_STATES)
      throw new PatternError(`pattern too large (>${MAX_STATES} states)`);
    states.push({ kind, cls, eps: [] });
    return states.length - 1;
  };
  const patch = (outs, t) => outs.forEach((s) => states[s].eps.push(t));

  const compile = (node) => {
    switch (node.type) {
      case "char": {
        const s = newState(KIND.CONSUME, node.cls);
        return { start: s, outs: [s] };
      }
      case "gate": {
        const s = newState(node.gate === "wb" ? KIND.GATE_WB : KIND.GATE_NWB);
        return { start: s, outs: [s] };
      }
      case "cat": {
        if (!node.parts.length) {
          const s = newState(KIND.EPS);
          return { start: s, outs: [s] };
        }
        const first = compile(node.parts[0]);
        let outs = first.outs;
        for (const p of node.parts.slice(1)) {
          const f = compile(p);
          patch(outs, f.start);
          outs = f.outs;
        }
        return { start: first.start, outs };
      }
      case "alt": {
        const frags = node.alts.map(compile);
        const s = newState(KIND.EPS);
        states[s].eps.push(...frags.map((f) => f.start));
        return { start: s, outs: frags.flatMap((f) => f.outs) };
      }
      case "rep": {
        const { q, a } = node;
        if (q === "?" || q === "*" || q === "+") {
          const f = compile(a);
          const s = newState(KIND.EPS);
          const e = newState(KIND.EPS);
          states[s].eps.push(f.start, e);
          if (q === "?") patch(f.outs, e); // '?' exits after one
          else patch(f.outs, s);           // '*'/'+' loop back (s already has exit edge)
          return { start: q === "+" ? f.start : s, outs: [e] };
        }
        // {n,m}: expand clones
        const [n0, m0] = q;
        if (n0 > 48 || (m0 !== Infinity && m0 > 48) || m0 < n0)
          throw new PatternError("bad {n,m} bounds");
        const cloneFrag = () => compile(a);
        let outs = null, start = null;
        const chain = (f) => {
          if (start === null) start = f.start;
          else patch(outs, f.start);
          outs = f.outs;
        };
        for (let k = 0; k < n0; k++) chain(cloneFrag());
        if (m0 === Infinity) {
          // trailing star
          const f = compile(a);
          const s = newState(KIND.EPS);
          const e = newState(KIND.EPS);
          states[s].eps.push(f.start, e);
          patch(f.outs, s);
          if (start === null) { start = s; outs = [e]; }
          else { patch(outs, s); outs = [e]; }
        } else {
          for (let k = n0; k < m0; k++) {
            const f = cloneFrag();
            const s = newState(KIND.EPS);
            const e = newState(KIND.EPS);
            states[s].eps.push(f.start, e);
            patch(f.outs, e);
            if (start === null) start = s;
            else patch(outs, s);
            outs = [e];
          }
        }
        if (start === null) { const s = newState(KIND.EPS); return { start: s, outs: [s] }; }
        return { start, outs };
      }
      default:
        throw new PatternError(`unknown node ${node.type}`);
    }
  };

  const frag = compile(ast);
  const acc = newState(KIND.ACCEPT);
  patch(frag.outs, acc);

  // fold char classes -> ids
  const key = (r) => (r === null ? "" : r.map((p) => p.join("-")).join(","));
  const classes = [[]]; // class 0 = any
  const clsId = new Map([["", 0]]);
  const kindA = new Uint8Array(states.length);
  const clsA = new Uint8Array(states.length);
  states.forEach((st, i) => {
    kindA[i] = st.kind;
    if (st.kind === KIND.CONSUME) {
      const r = st.cls === null ? null : normRanges(foldRanges(st.cls, icase));
      const k = key(r);
      if (!clsId.has(k)) {
        if (classes.length >= MAX_CLASSES)
          throw new PatternError(`too many char classes (>${MAX_CLASSES})`);
        clsId.set(k, classes.length);
        classes.push(r);
      }
      clsA[i] = clsId.get(k);
    }
  });

  const S = states.length;
  const bit = (m, s) => { m[s >> 5] |= 1 << (s & 31); };

  // eps0 closure masks (EPS edges only; gates are barriers)
  const eps0 = new Uint32Array(S * 2);
  for (let s = 0; s < S; s++) {
    const seen = new Uint32Array(2);
    const stack = [s];
    while (stack.length) {
      const t = stack.pop();
      if (seen[t >> 5] & (1 << (t & 31))) continue;
      bit(seen, t);
      if (states[t].kind === KIND.EPS) stack.push(...states[t].eps);
    }
    eps0[s * 2] = seen[0]; eps0[s * 2 + 1] = seen[1];
  }
  // fire[s] = eps0 of a CONSUME/GATE state's (single) target
  const fire = new Uint32Array(S * 2);
  for (let s = 0; s < S; s++) {
    if (states[s].kind === KIND.CONSUME || states[s].kind === KIND.GATE_WB || states[s].kind === KIND.GATE_NWB) {
      const m = [0, 0];
      for (const t of states[s].eps) { m[0] |= eps0[t * 2]; m[1] |= eps0[t * 2 + 1]; }
      fire[s * 2] = m[0]; fire[s * 2 + 1] = m[1];
    }
  }
  const consumeMask = [0, 0], gateMask = [0, 0], acceptMask = [0, 0];
  let hasGates = false;
  states.forEach((st, i) => {
    if (st.kind === KIND.CONSUME) bit(consumeMask, i);
    else if (st.kind === KIND.GATE_WB || st.kind === KIND.GATE_NWB) { bit(gateMask, i); hasGates = true; }
    else if (st.kind === KIND.ACCEPT) bit(acceptMask, i);
  });

  // cpBmp: 65536-entry u32, bit c = cp in class c (bit 0 always set: ".")
  const cpBmp = new Uint32Array(65536);
  const inRanges = (rs, cp) => {
    if (rs === null) return true;
    for (const [a, b] of rs) if (cp >= a && cp <= b) return true;
    return false;
  };
  for (let cp = 0; cp < 65536; cp++) {
    let m = 1; // class 0: any
    for (let c = 1; c < classes.length; c++)
      if (inRanges(classes[c], cp)) m |= 1 << c;
    cpBmp[cp] = m;
  }
  // astral range table
  const clsRanges = [], clsRangeOff = new Uint32Array(classes.length + 1);
  for (let c = 0; c < classes.length; c++) {
    clsRangeOff[c] = clsRanges.length;
    const rs = classes[c];
    if (rs) for (const [a, b] of rs) if (b > 0xffff) clsRanges.push(Math.max(a, 0x10000), b);
  }
  clsRangeOff[classes.length] = clsRanges.length;

  return {
    kind: kindA, cls: clsA, fire, eps0,
    start: [eps0[frag.start * 2], eps0[frag.start * 2 + 1]],
    consumeMask, gateMask, acceptMask,
    cpBmp, clsRanges: Uint32Array.from(clsRanges), clsRangeOff,
    nStates: S, nClasses: classes.length, hasGates, ignoreCase: icase,
  };
}

/** Compile a pattern. Returns {tables, error}. `flags`: 'i' for ignorecase. */
export function compilePattern(src, flags = "") {
  if (!src) return { tables: null, error: null };
  const icase = flags.includes("i");
  try {
    const ast = parse(src);
    return { tables: build(ast, icase), error: null };
  } catch (e) {
    return { tables: null, error: e instanceof PatternError ? e.message : String(e) };
  }
}

/** Escape regex metas so a plain-word query searches literally. */
export function literalize(src) {
  return src.replace(/[\\^$.*+?()[\]{}|]/g, (c) => "\\" + c);
}
