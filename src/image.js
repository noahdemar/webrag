// Typed-array image ops mirroring the cv2 calls in indra_ocr.
// Grayscale pages are Uint8Array (or Uint8ClampedArray) [h*w], row-major.

/** RGBA ImageData -> grayscale Uint8Array (ITU-R 601 luma, like pdfium grayscale). */
export function rgbaToGray(rgba, w, h) {
  const n = w * h;
  const out = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    // canvas alpha is over transparent black; composite onto white paper first
    const a = rgba[j + 3] / 255;
    const r = rgba[j] * a + 255 * (1 - a);
    const g = rgba[j + 1] * a + 255 * (1 - a);
    const b = rgba[j + 2] * a + 255 * (1 - a);
    out[i] = (77 * r + 150 * g + 29 * b) >> 8;
  }
  return out;
}

/** Bilinear resize of a gray plane to (nw, nh) — cv2.INTER_LINEAR. */
export function resizeBilinear(gray, w, h, nw, nh) {
  const out = new Uint8Array(nw * nh);
  const sx = w / nw, sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const fy = (y + 0.5) * sy - 0.5;
    const y0 = Math.floor(fy), wy = fy - y0;
    const ya = Math.max(0, Math.min(h - 1, y0));
    const yb = Math.max(0, Math.min(h - 1, y0 + 1));
    for (let x = 0; x < nw; x++) {
      const fx = (x + 0.5) * sx - 0.5;
      const x0 = Math.floor(fx), wx = fx - x0;
      const xa = Math.max(0, Math.min(w - 1, x0));
      const xb = Math.max(0, Math.min(w - 1, x0 + 1));
      const p00 = gray[ya * w + xa], p01 = gray[ya * w + xb];
      const p10 = gray[yb * w + xa], p11 = gray[yb * w + xb];
      const p = (p00 * (1 - wx) + p01 * wx) * (1 - wy) + (p10 * (1 - wx) + p11 * wx) * wy;
      out[y * nw + x] = p < 0 ? 0 : p > 255 ? 255 : p | 0;
    }
  }
  return out;
}

/** Area-average downscale — cv2.INTER_AREA (used when shrinking). */
export function resizeArea(gray, w, h, nw, nh) {
  const out = new Uint8Array(nw * nh);
  const sx = w / nw, sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const y0 = y * sy, y1 = y0 + sy;
    const iy0 = Math.floor(y0), iy1 = Math.ceil(y1);
    for (let x = 0; x < nw; x++) {
      const x0 = x * sx, x1 = x0 + sx;
      const ix0 = Math.floor(x0), ix1 = Math.ceil(x1);
      let sum = 0, area = 0;
      for (let yy = iy0; yy < iy1; yy++) {
        const cy = Math.min(y1, yy + 1) - Math.max(y0, yy);
        if (cy <= 0) continue;
        const row = yy * w;
        for (let xx = ix0; xx < ix1; xx++) {
          const cx = Math.min(x1, xx + 1) - Math.max(x0, xx);
          if (cx <= 0) continue;
          sum += gray[row + xx] * cx * cy;
          area += cx * cy;
        }
      }
      out[y * nw + x] = area > 0 ? (sum / area) | 0 : 255;
    }
  }
  return out;
}

/** cv2.resize choosing INTER_AREA down / INTER_LINEAR up, like the pipeline. */
export function resizeGray(gray, w, h, nw, nh) {
  if (nw === w && nh === h) return gray;
  return (nw * nh < w * h ? resizeArea : resizeBilinear)(gray, w, h, nw, nh);
}

/** Float32 gray plane bilinear resize (mask output path). */
export function resizeBilinearF(f, w, h, nw, nh) {
  const out = new Float32Array(nw * nh);
  const sx = w / nw, sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const fy = (y + 0.5) * sy - 0.5;
    const y0 = Math.floor(fy), wy = fy - y0;
    const ya = Math.max(0, Math.min(h - 1, y0));
    const yb = Math.max(0, Math.min(h - 1, y0 + 1));
    for (let x = 0; x < nw; x++) {
      const fx = (x + 0.5) * sx - 0.5;
      const x0 = Math.floor(fx), wx = fx - x0;
      const xa = Math.max(0, Math.min(w - 1, x0));
      const xb = Math.max(0, Math.min(w - 1, x0 + 1));
      const p00 = f[ya * w + xa], p01 = f[ya * w + xb];
      const p10 = f[yb * w + xa], p11 = f[yb * w + xb];
      out[y * nw + x] = (p00 * (1 - wx) + p01 * wx) * (1 - wy) + (p10 * (1 - wx) + p11 * wx) * wy;
    }
  }
  return out;
}

/** np.rot90(-k): k quarter turns clockwise (k=1 -> 90° CW). Returns {data,w,h}. */
export function rot90cw(gray, w, h, k) {
  k = ((k % 4) + 4) % 4;
  if (k === 0) return { data: gray, w, h };
  let nw = k % 2 ? h : w, nh = k % 2 ? w : h;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let nx, ny;
      if (k === 1) { nx = h - 1 - y; ny = x; }
      else if (k === 2) { nx = w - 1 - x; ny = h - 1 - y; }
      else { nx = y; ny = w - 1 - x; }
      out[ny * nw + nx] = gray[row + x];
    }
  }
  return { data: out, w: nw, h: nh };
}

/** cv2.getRotationMatrix2D + warpAffine(bilinear, BORDER_REPLICATE): rotate about
 *  centre by `deg` counter-clockwise (positive = CCW), same output size. */
export function rotatePage(gray, w, h, deg) {
  const rad = (deg * Math.PI) / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const cx = w / 2, cy = h / 2;
  // inverse map: for each dst pixel, source location
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const dy = y - cy;
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      // dst = R(+deg) src  =>  src = R(-deg) dst
      const sx = cos * dx + sin * dy + cx;
      const sy = -sin * dx + cos * dy + cy;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const wx = sx - x0, wy = sy - y0;
      // BORDER_REPLICATE: clamp
      const xa = Math.max(0, Math.min(w - 1, x0));
      const xb = Math.max(0, Math.min(w - 1, x0 + 1));
      const ya = Math.max(0, Math.min(h - 1, y0));
      const yb = Math.max(0, Math.min(h - 1, y0 + 1));
      const p00 = gray[ya * w + xa], p01 = gray[ya * w + xb];
      const p10 = gray[yb * w + xa], p11 = gray[yb * w + xb];
      out[y * w + x] =
        (p00 * (1 - wx) + p01 * wx) * (1 - wy) + (p10 * (1 - wx) + p11 * wx) * wy;
    }
  }
  return out;
}

const DESKEW_MAX = 5.0, DESKEW_MIN = 0.3;

/** Degrees (CCW) that straighten the text rows — port of pipeline.estimate_skew. */
export function estimateSkew(gray, w, h, side = 800) {
  const s = side / Math.max(h, w);
  const small = resizeArea(gray, w, h, Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s)));
  const sw = Math.max(1, Math.round(w * s)), sh = Math.max(1, Math.round(h * s));
  const ink = new Float32Array(sw * sh);
  let mean = 0;
  for (let i = 0; i < ink.length; i++) { ink[i] = (255 - small[i]) / 255; mean += ink[i]; }
  mean /= ink.length;
  if (mean < 0.003) return 0.0;

  // rotate a float ink map (bilinear, replicate border), then row-profile sharpness
  const rotF = (a) => {
    const rad = (a * Math.PI) / 180, cos = Math.cos(rad), sin = Math.sin(rad);
    const cx = sw / 2, cy = sh / 2;
    const out = new Float32Array(sw * sh);
    for (let y = 0; y < sh; y++) {
      const dy = y - cy;
      for (let x = 0; x < sw; x++) {
        const dx = x - cx;
        const sx = cos * dx + sin * dy + cx;
        const sy = -sin * dx + cos * dy + cy;
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const wx = sx - x0, wy = sy - y0;
        const xa = Math.max(0, Math.min(sw - 1, x0));
        const xb = Math.max(0, Math.min(sw - 1, x0 + 1));
        const ya = Math.max(0, Math.min(sh - 1, y0));
        const yb = Math.max(0, Math.min(sh - 1, y0 + 1));
        const p00 = ink[ya * sw + xa], p01 = ink[ya * sw + xb];
        const p10 = ink[yb * sw + xa], p11 = ink[yb * sw + xb];
        out[y * sw + x] = (p00 * (1 - wx) + p01 * wx) * (1 - wy) + (p10 * (1 - wx) + p11 * wx) * wy;
      }
    }
    return out;
  };
  const sharpness = (a) => {
    const r = rotF(a);
    let acc = 0;
    for (let y = 1; y < sh; y++) {
      let d = 0, p0 = 0, p1 = 0;
      for (let x = 0; x < sw; x++) { p0 += r[(y - 1) * sw + x]; p1 += r[y * sw + x]; }
      d = p1 - p0; acc += d * d;
    }
    return acc;
  };
  let best = -DESKEW_MAX, bestS = -Infinity;
  for (let a = -DESKEW_MAX; a <= DESKEW_MAX + 1e-6; a += 0.5) {
    const s2 = sharpness(a);
    if (s2 > bestS) { bestS = s2; best = a; }
  }
  let best2 = best, bestS2 = -Infinity;
  for (let a = best - 0.5; a <= best + 0.5 + 1e-6; a += 0.1) {
    const s2 = sharpness(a);
    if (s2 > bestS2) { bestS2 = s2; best2 = a; }
  }
  return Math.abs(best2) >= DESKEW_MIN ? best2 : 0;
}

/** The page rotated so that `deg` CCW tilt is undone. */
export function deskew(gray, w, h, deg) {
  return rotatePage(gray, w, h, deg);
}

/** Map a box on the deskewed page back to the turned page (pipeline.unrotate_box). */
export function unrotateBox(box, deg, w, h) {
  const rad = (deg * Math.PI) / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const cx = w / 2, cy = h / 2;
  const [x0, y0, x1, y1] = box;
  const xs = [x0, x1, x0, x1], ys = [y0, y0, y1, y1];
  let nx0 = Infinity, ny0 = Infinity, nx1 = -Infinity, ny1 = -Infinity;
  for (let i = 0; i < 4; i++) {
    const dx = xs[i] - cx, dy = ys[i] - cy;
    const mx = cos * dx - sin * dy + cx;
    const my = sin * dx + cos * dy + cy;
    nx0 = Math.min(nx0, mx); nx1 = Math.max(nx1, mx);
    ny0 = Math.min(ny0, my); ny1 = Math.max(ny1, my);
  }
  return [
    Math.round(Math.max(0, Math.min(w, nx0)) * 10) / 10,
    Math.round(Math.max(0, Math.min(h, ny0)) * 10) / 10,
    Math.round(Math.max(0, Math.min(w, nx1)) * 10) / 10,
    Math.round(Math.max(0, Math.min(h, ny1)) * 10) / 10,
  ];
}

/** median of a numeric array (small helper, sorts a copy). */
export function median(a) {
  if (!a.length) return 0;
  const s = Float64Array.from(a).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
