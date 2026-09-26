/* In-browser version of the CrossWatch pipeline for the live demo.
   Runs entirely on the visitor's machine: the video never leaves the browser.

   Same scene geometry (demo/scene.json, exported from src/scene.py), same rules and
   thresholds as src/rules.py, same risk model as src/risk.py. Differences from the
   submission, all for speed in a browser: YOLO11n instead of YOLO11m, a simplified
   ByteTrack-style tracker, translation-only alignment to the reference view, and
   wrong_way is not evaluated (it needs the learned flow field). */
"use strict";

const Demo = (() => {
  const SAMPLE_FPS = 5, DT = 1 / SAMPLE_FPS;
  const IN_W = 960, IN_H = 544, FRAME_H = 540;          // model input; frame is letterboxed to 960x540
  const PERSON = 0, BICYCLE = 1, VEHICLES = new Set([2, 3, 5, 7]), ANIMALS = new Set([15, 16, 17, 18, 19]);
  const KEEP = new Set([0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19]);
  const NAMES = { 0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow" };
  const STILL = 0.12, MOVING = 0.5, RED = 1, GREEN = 0, UNKNOWN = -1;
  const MAX_SECONDS = 180;

  let scene = null, session = null, refGray = null;

  /* ---------------- loading ---------------- */
  async function init(progress) {
    if (!scene) scene = await (await fetch("demo/scene.json")).json();
    if (!session) {
      progress("loading the detector (11 MB, first time only)", 0.01);
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);   // >1 only where the page is cross-origin isolated
      try {
        session = await ort.InferenceSession.create("demo/yolo11n_960x544.onnx", { executionProviders: ["webgpu", "wasm"] });
      } catch (e) {
        session = await ort.InferenceSession.create("demo/yolo11n_960x544.onnx", { executionProviders: ["wasm"] });
      }
    }
    if (!refGray) {
      const img = new Image(); img.src = "demo/scene_ref.jpg"; await img.decode();
      refGray = grayGradient(img, 320, 180);
    }
  }

  /* ---------------- video helpers ---------------- */
  function seek(video, t) {
    return new Promise((resolve) => {
      const done = () => { clearTimeout(timer); video.removeEventListener("seeked", done); resolve(); };
      const timer = setTimeout(done, 3000);
      video.addEventListener("seeked", done);
      video.currentTime = Math.min(t, Math.max(0, video.duration - 0.01));
    });
  }

  function grayGradient(src, w, h) {
    const c = new OffscreenCanvas(w, h), ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(src, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data, g = new Float32Array(w * h), out = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out[i] = Math.abs(g[i + 1] - g[i - 1]) + Math.abs(g[i + w] - g[i - w]);
    }
    return out;
  }

  /* Translation between the uploaded view and the reference view (normalised units, video -> ref). */
  async function align(video) {
    const W = 320, H = 180, n = 5, frames = [];
    for (let k = 0; k < n; k++) { await seek(video, ((k + 0.5) / n) * video.duration); frames.push(grayGradient(video, W, H)); }
    const med = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) med[i] = frames.map((f) => f[i]).sort((a, b) => a - b)[n >> 1];
    let best = [0, 0], bestErr = Infinity;
    const R = 20, M = 24;
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      let err = 0;
      for (let y = M; y < H - M; y += 2) for (let x = M; x < W - M; x += 2) err += Math.abs(med[y * W + x] - refGray[(y + dy) * W + x + dx]);
      if (err < bestErr) { bestErr = err; best = [dx, dy]; }
    }
    return { dx: best[0] / W, dy: best[1] / H };
  }

  /* ---------------- detection ---------------- */
  const inCanvas = new OffscreenCanvas(IN_W, IN_H);
  const inCtx = inCanvas.getContext("2d", { willReadFrequently: true });

  async function detect(video) {
    inCtx.fillStyle = "rgb(114,114,114)"; inCtx.fillRect(0, 0, IN_W, IN_H);
    inCtx.drawImage(video, 0, 0, IN_W, FRAME_H);
    const px = inCtx.getImageData(0, 0, IN_W, IN_H).data, N = IN_W * IN_H;
    const input = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { input[i] = px[4 * i] / 255; input[N + i] = px[4 * i + 1] / 255; input[2 * N + i] = px[4 * i + 2] / 255; }
    const out = await session.run({ images: new ort.Tensor("float32", input, [1, 3, IN_H, IN_W]) });
    const data = out.output0.data, A = out.output0.dims[2];
    const cand = [];
    for (let i = 0; i < A; i++) {
      let best = 0, cls = -1;
      for (const c of KEEP) { const s = data[(4 + c) * A + i]; if (s > best) { best = s; cls = c; } }
      if (best < 0.1) continue;
      const cx = data[i], cy = data[A + i], w = data[2 * A + i], h = data[3 * A + i];
      cand.push([(cx - w / 2) / IN_W, (cy - h / 2) / FRAME_H, (cx + w / 2) / IN_W, (cy + h / 2) / FRAME_H, best, cls]);
    }
    return nms(cand, 0.5);
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy, u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return u > 0 ? inter / u : 0;
  }

  function nms(boxes, thr) {
    boxes.sort((a, b) => b[4] - a[4]);
    const keep = [];
    for (const b of boxes) {
      if (keep.length >= 300) break;
      if (keep.every((k) => k[5] !== b[5] || iou(k, b) < thr)) keep.push(b);
    }
    return keep;
  }

  /* ---------------- tracking (ByteTrack-style, two-stage IoU matching) ---------------- */
  class Tracker {
    constructor() { this.tracks = []; this.next = 1; }
    update(dets, t) {
      for (const tr of this.tracks) {                     // constant-velocity prediction
        const dt = t - tr.t;
        tr.pred = [tr.box[0] + tr.v[0] * dt, tr.box[1] + tr.v[1] * dt, tr.box[2] + tr.v[0] * dt, tr.box[3] + tr.v[1] * dt];
      }
      const high = dets.filter((d) => d[4] >= 0.35), low = dets.filter((d) => d[4] < 0.35);
      const free = new Set(this.tracks), out = [];
      const match = (ds, minIou) => {
        const pairs = [];
        for (const d of ds) for (const tr of free) { const o = iou(d, tr.pred); if (o >= minIou) pairs.push([o, d, tr]); }
        pairs.sort((a, b) => b[0] - a[0]);
        const usedD = new Set();
        for (const [, d, tr] of pairs) {
          if (usedD.has(d) || !free.has(tr)) continue;
          usedD.add(d); free.delete(tr);
          const dt = Math.max(1e-3, t - tr.t);
          const vx = ((d[0] + d[2]) - (tr.box[0] + tr.box[2])) / 2 / dt, vy = (d[3] - tr.box[3]) / dt;
          tr.v = [0.5 * tr.v[0] + 0.5 * vx, 0.5 * tr.v[1] + 0.5 * vy];
          tr.box = d.slice(0, 4); tr.t = t; tr.votes[d[5]] = (tr.votes[d[5]] || 0) + 1;
          out.push([t, tr.id, d[5], d[4], ...d.slice(0, 4)]);
        }
        return ds.filter((d) => !usedD.has(d));
      };
      const leftHigh = match(high, 0.2);
      match(low, 0.5);
      for (const d of leftHigh) {
        if (d[4] < 0.4) continue;
        const tr = { id: this.next++, box: d.slice(0, 4), v: [0, 0], t, votes: { [d[5]]: 1 } };
        this.tracks.push(tr);
        out.push([t, tr.id, d[5], d[4], ...d.slice(0, 4)]);
      }
      this.tracks = this.tracks.filter((tr) => t - tr.t <= 5);
      return out;
    }
  }

  /* ---------------- traffic-signal lamps ---------------- */
  const lampCanvas = new OffscreenCanvas(48, 96);
  const lampCtx = lampCanvas.getContext("2d", { willReadFrequently: true });

  function lampScores(video, shift) {
    let red = 0, green = 0, total = 0;
    const vw = video.videoWidth, vh = video.videoHeight;
    for (const [x1, y1, x2, y2] of scene.signal_heads) {
      const bx1 = Math.max(0, x1 - shift.dx - 0.004), by1 = Math.max(0, y1 - shift.dy - 0.004);
      const bx2 = Math.min(1, x2 - shift.dx + 0.004), by2 = Math.min(1, y2 - shift.dy + 0.004);
      lampCtx.drawImage(video, bx1 * vw, by1 * vh, (bx2 - bx1) * vw, (by2 - by1) * vh, 0, 0, 48, 96);
      const d = lampCtx.getImageData(0, 0, 48, 96).data;
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i + 1], b = d[i + 2], v = Math.max(r, g, b), mn = Math.min(r, g, b);
        total++;
        if (v <= 110 || v === 0) continue;
        const s = ((v - mn) * 255) / v;
        if (s <= 80) continue;
        let h = v === r ? (60 * (g - b)) / (v - mn) : v === g ? 120 + (60 * (b - r)) / (v - mn) : 240 + (60 * (r - g)) / (v - mn);
        if (h < 0) h += 360;
        h /= 2;                                            // OpenCV hue scale 0-180
        if (h < 10 || h > 168) red++; else if (h > 55 && h < 100) green++;
      }
    }
    return [red / total, green / total];
  }

  function stateTimeline(scores, minHold = 5) {
    const raw = scores.map(([r, g]) => (Math.max(r, g) < 0.0015 ? UNKNOWN : r > g ? RED : GREEN));
    const known = raw.filter((s) => s !== UNKNOWN);
    let cur = known.length ? known[0] : UNKNOWN, runVal = cur, runLen = 0;
    const out = new Array(raw.length);
    raw.forEach((s, i) => {
      if (s !== UNKNOWN && s !== cur) {
        runLen = s === runVal ? runLen + 1 : 1; runVal = s;
        if (runLen >= minHold) { cur = s; for (let k = i - minHold + 1; k < i; k++) out[k] = s; }
      } else runLen = 0;
      out[i] = cur;
    });
    return out;
  }

  /* ---------------- accident risk (causal, same model as src/risk.py) ---------------- */
  class Risk {
    constructor() { this.hist = new Map(); this.cls = new Map(); this.score = 0; this.lastT = 0; }
    step(rows, t) {
      for (const [, id, c, , x1, y1, x2, y2] of rows) {
        if (!this.hist.has(id)) this.hist.set(id, []);
        const h = this.hist.get(id); h.push([t, (x1 + x2) / 2, y2, x2 - x1, y2 - y1]);
        if (h.length > 12) h.shift();
        this.cls.set(id, c);
      }
      for (const [id, h] of this.hist) if (t - h[h.length - 1][0] > 2) this.hist.delete(id);
      const states = [];
      for (const [, id] of rows) { const s = this.state(id); if (s) states.push([id, s]); }
      let conflict = 0, brake = 0;
      for (let i = 0; i < states.length; i++) {
        const [ia, a] = states[i], vehA = VEHICLES.has(this.cls.get(ia));
        if (vehA) brake = Math.max(brake, Math.min(1, Math.max(0, a.decel - 2.5) / 3));
        for (let j = i + 1; j < states.length; j++) {
          const [ib, b] = states[j];
          if (!(vehA || VEHICLES.has(this.cls.get(ib)))) continue;
          if (Math.hypot(a.x - b.x, a.y - b.y) > 0.25) continue;
          const hit = this.ttc(a, b);
          if (hit) conflict = Math.max(conflict, Math.exp(-hit[0] / 0.8) * Math.min(1, hit[1] / 3));
        }
      }
      const hz = Math.max(conflict, 0.8 * brake, 0.5 * (conflict + brake));
      const p = 1 / (1 + Math.exp(-10 * (hz - 0.72)));
      this.score = Math.max(p, this.score * Math.pow(0.5, t - this.lastT));
      this.lastT = t;
      return this.score;
    }
    state(id) {
      const h = this.hist.get(id);
      if (!h || h.length < 3 || h[h.length - 1][0] - h[0][0] < 0.3) return null;
      const n = h.length, last = h[n - 1], k = Math.max(0, n - 1 - 3), dt = Math.max(1e-3, last[0] - h[k][0]);
      const vx = (last[1] - h[k][1]) / dt, vy = (last[2] - h[k][2]) / dt;
      const hh = Math.max(1e-3, median(h.map((r) => r[4]))), speed = Math.hypot(vx, vy) / hh;
      let decel = 0;
      if (last[0] - h[0][0] >= 1.2) {
        let j = h.findIndex((r) => r[0] >= last[0] - 1.2); if (j < 0) j = 0;
        const j2 = Math.min(j + 3, n - 1), dt0 = Math.max(1e-3, h[j2][0] - h[j][0]);
        decel = Math.max(0, Math.hypot(h[j2][1] - h[j][1], h[j2][2] - h[j][2]) / dt0 / hh - speed);
      }
      return { x: last[1], y: last[2], w: median(h.slice(-3).map((r) => r[3])), h: hh, vx, vy, speed, decel };
    }
    ttc(a, b) {
      if (Math.max(a.speed, b.speed) < 1.0) return null;
      const rx = b.x - a.x, ry = b.y - a.y, rvx = b.vx - a.vx, rvy = b.vy - a.vy, dist = Math.max(1e-6, Math.hypot(rx, ry));
      const closing = -(rx * rvx + ry * rvy) / dist / Math.max(1e-3, Math.min(a.h, b.h));
      if (closing < 0.8) return null;
      const fp = (s, o) => [o.x + o.vx * s - o.w / 2, o.y + o.vy * s - 0.3 * o.h, o.x + o.vx * s + o.w / 2, o.y + o.vy * s];
      const over = (p, q) => p[0] < q[2] && q[0] < p[2] && p[1] < q[3] && q[1] < p[3];
      if (over(fp(0, a), fp(0, b))) return null;
      for (let s = 0.1; s <= 3.0001; s += 0.1) if (over(fp(s, a), fp(s, b))) return [s, closing];
      return null;
    }
  }

  /* ---------------- scene queries ---------------- */
  function inPoly(poly, x, y) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  const inCrosswalk = (x, y, strict) => Object.entries(strict ? scene.stripes : scene.crosswalks).find(([, p]) => inPoly(p, x, y))?.[0] ?? null;
  function onRoad(x, y) {
    const [GW, GH] = scene.grid;
    const r = Math.min(GH - 1, Math.max(0, Math.floor(y * GH))), c = Math.min(GW - 1, Math.max(0, Math.floor(x * GW)));
    return scene.road_core[r * GW + c] === 1 && !scene.non_road.some((p) => inPoly(p, x, y));
  }
  const zoneOf = (x, y) => Object.entries(scene.zones).find(([, z]) => inPoly(z.poly, x, y))?.[0] ?? null;
  const inParking = (x, y) => scene.curb_parking.some((p) => inPoly(p, x, y));
  function stopLineSide(x, y) { const [[x1, y1], [x2, y2]] = scene.stop_line; return y - (y1 + ((y2 - y1) * (x - x1)) / (x2 - x1)); }
  function nearStopLineSpan(x, m = 0.02) { const [[x1], [x2]] = scene.stop_line; return x1 - m <= x && x <= x2 + m; }

  /* ---------------- trajectories (as src/tracks.py) ---------------- */
  const median = (a) => { const s = [...a].sort((p, q) => p - q); return s.length ? s[s.length >> 1] : 0; };
  function smooth(a, k = 3) {
    if (a.length < k) return a.slice();
    const pad = k >> 1, p = [...Array(pad).fill(a[0]), ...a, ...Array(pad).fill(a[a.length - 1])];
    return a.map((_, i) => { let s = 0; for (let j = 0; j < k; j++) s += p[i + j]; return s / k; });
  }
  function velocity(t, p, half = 0.6) {
    return t.map((ti) => {
      let lo = t.findIndex((x) => x >= ti - half); let hi = t.length - 1;
      while (hi > 0 && t[hi] > ti + half) hi--;
      if (lo < 0) lo = 0;
      const dt = t[hi] - t[lo];
      return dt > 1e-6 ? (p[hi] - p[lo]) / dt : 0;
    });
  }
  function buildTracks(rows) {
    const by = new Map();
    for (const r of rows) { if (!by.has(r[1])) by.set(r[1], []); by.get(r[1]).push(r); }
    const tracks = [];
    for (const [tid, rs] of by) {
      if (rs.length < 3) continue;
      rs.sort((a, b) => a[0] - b[0]);
      const votes = {}; rs.forEach((r) => (votes[r[2]] = (votes[r[2]] || 0) + 1));
      const cls = +Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
      const t = rs.map((r) => r[0]);
      const tr = { tid, cls, t, x: smooth(rs.map((r) => (r[4] + r[6]) / 2)), y: smooth(rs.map((r) => r[7])),
        w: smooth(rs.map((r) => r[6] - r[4])), h: smooth(rs.map((r) => r[7] - r[5])), conf: rs.map((r) => r[3]) };
      tr.vx = velocity(t, tr.x); tr.vy = velocity(t, tr.y);
      tr.speed = tr.vx.map((vx, i) => Math.hypot(vx, tr.vy[i]) / Math.max(tr.h[i], 1e-3));
      tr.isVehicle = VEHICLES.has(cls); tr.isPerson = cls === PERSON;
      tr.at = (tm) => { let best = null; t.forEach((ti, i) => { if (Math.abs(ti - tm) <= 0.6 * DT && (best === null || Math.abs(ti - tm) < Math.abs(t[best] - tm))) best = i; }); return best; };
      tracks.push(tr);
    }
    return tracks;
  }

  /* ---------------- segments (as src/segments.py) ---------------- */
  function runs(mask) {
    const out = []; let s = null;
    mask.forEach((m, i) => { if (m && s === null) s = i; else if (!m && s !== null) { out.push([s, i - 1]); s = null; } });
    if (s !== null) out.push([s, mask.length - 1]);
    return out;
  }
  function fillGaps(mask, maxGap) {
    const m = mask.slice();
    for (const [i, j] of runs(mask.map((x) => !x))) if (i > 0 && j < mask.length - 1 && j - i + 1 <= maxGap) for (let k = i; k <= j; k++) m[k] = true;
    return m;
  }
  function merge(segs, gap) {
    const out = [];
    for (const [s, e] of [...segs].sort((a, b) => a[0] - b[0])) {
      if (out.length && s - out[out.length - 1][1] <= gap) out[out.length - 1][1] = Math.max(out[out.length - 1][1], e);
      else out.push([s, e]);
    }
    return out;
  }
  const clean = (segs, gap, minLen, dur) => merge(segs, gap).map(([s, e]) => [Math.max(0, s), Math.min(dur, e)])
    .filter(([s, e]) => e - s >= minLen).map(([s, e]) => [+s.toFixed(2), +e.toFixed(2)]);
  function flagsToSegs(times, flags) {
    return runs(flags).map(([i, j]) => [times[i], times[j] + DT]);
  }

  /* ---------------- rules (as src/rules.py) ---------------- */
  function rules(tracks, times, signal, dur) {
    const sigAt = (t) => { let i = times.findIndex((x) => x >= t); if (i < 0) i = times.length - 1; return signal[i]; };
    const vehicles = tracks.filter((t) => t.isVehicle), people = tracks.filter((t) => t.isPerson);
    const reliable = (p, k) => p.h[k] >= 0.035 && p.conf[k] >= 0.35 && p.y[k] < 0.985;   // conf 0.45 in Python; YOLO11n scores lower
    const ev = {};

    // jaywalking
    let segs = [];
    for (const p of people) {
      const on = fillGaps(p.x.map((x, k) => reliable(p, k) && onRoad(x, p.y[k]) && inCrosswalk(x, p.y[k], false) === null), 3);
      for (const [i, j] of runs(on)) {
        const d = p.t[j] - p.t[i], hh = Math.max(median(p.h.slice(i, j + 1)), 1e-3);
        const moved = Math.hypot(p.x[j] - p.x[i], p.y[j] - p.y[i]) / hh, sp = p.speed.slice(i, j + 1).sort((a, b) => a - b);
        if (d >= 1.5 && d <= 60 && moved >= 1 && median(sp) < 2.5 && sp[Math.floor(0.9 * (sp.length - 1))] < 4) segs.push([p.t[i], p.t[j] + DT]);
      }
    }
    ev.jaywalking = clean(segs, 2, 1.5, dur);

    // failure_to_yield
    const peds = {};
    for (const p of people) p.t.forEach((t, k) => {
      if (!reliable(p, k) || p.speed[k] < 0.3) return;
      const cw = inCrosswalk(p.x[k], p.y[k], true);
      if (cw && onRoad(p.x[k], p.y[k])) (peds[cw] = peds[cw] || []).push([t, p.x[k], p.y[k]]);
    });
    segs = [];
    for (const v of vehicles) {
      const inside = v.x.map((x, k) => inCrosswalk(x, v.y[k], false));
      for (const name of new Set(inside.filter(Boolean))) {
        const mask = fillGaps(inside.map((c, k) => c === name && v.speed[k] > MOVING), 2);
        for (const [i, j] of runs(mask)) {
          if (!peds[name] || v.t[j] - v.t[i] < 0.4) continue;
          let conflict = false;
          for (let k = i; k <= j && !conflict; k++) conflict = peds[name].some(([t, x, y]) =>
            Math.abs(t - v.t[k]) <= DT / 2 && Math.hypot(x - v.x[k], (y - v.y[k]) * 0.6) < Math.max(0.06, 1.2 * v.w[k]));
          if (conflict) segs.push([v.t[i], v.t[j] + DT]);
        }
      }
    }
    ev.failure_to_yield = clean(segs, 1, 0.6, dur);

    // red_light
    const crossing = (v) => {
      const side = v.x.map((x, k) => stopLineSide(x, v.y[k]));
      for (let k = 1; k < side.length; k++) {
        if (side[k - 1] < 0 && side[k] >= 0 && nearStopLineSpan(v.x[k]) && v.vy[k] > 0 && v.speed[k] > MOVING) {
          const later = side.slice(k, k + 11);
          if (side[Math.max(0, k - 5)] < -0.005 && Math.max(...later) > 0.03) return k;
        }
      }
      return null;
    };
    segs = [];
    for (const v of vehicles) {
      const k = crossing(v); if (k === null) continue;
      const tc = v.t[k];
      if (![tc - 4, tc, tc + 1].every((t) => sigAt(t) === RED)) continue;
      let end = v.t[v.t.length - 1] + DT;
      for (let m = k; m < v.t.length; m++) if (v.y[m] > 0.85 || v.x[m] > 0.95) { end = v.t[m]; break; }
      segs.push([tc, Math.min(end, tc + 8)]);
    }
    ev.red_light = clean(segs, 0, 0.8, dur);

    // stop_line
    segs = [];
    for (const v of vehicles) {
      for (let k = 0; k < v.t.length; k++) {
        const s = stopLineSide(v.x[k], v.y[k]);
        if (!(s > 0.004 && s < 0.09 && nearStopLineSpan(v.x[k], 0))) continue;
        if (v.speed[k] > STILL || sigAt(v.t[k]) !== RED) continue;
        let i0 = k; while (i0 > 0 && v.speed[i0 - 1] <= STILL) i0--;
        if (sigAt(v.t[i0]) !== RED || i0 === 0) break;
        let j = k; while (j + 1 < v.t.length && v.speed[j + 1] <= STILL) j++;
        if (v.t[j] - v.t[k] < 2) continue;
        let tg = v.t[k]; while (tg < dur && sigAt(tg) === RED) tg += DT;
        segs.push([v.t[k], tg]); break;
      }
    }
    ev.stop_line = clean(segs, 1, 2, dur);

    // congestion, per zone
    const zoneNames = Object.keys(scene.zones), jams = {}, win = 3 * SAMPLE_FPS;
    const idx = new Map(times.map((t, i) => [t.toFixed(3), i]));
    const count = zoneNames.map(() => new Float32Array(times.length)), still = zoneNames.map(() => new Float32Array(times.length));
    for (const v of vehicles) v.t.forEach((t, k) => {
      const z = zoneOf(v.x[k], v.y[k]), i = idx.get(t.toFixed(3));
      if (z === null || i === undefined) return;
      const zi = zoneNames.indexOf(z); count[zi][i]++; if (v.speed[k] < STILL * 1.5) still[zi][i]++;
    });
    const box = (a) => a.map((_, i) => { let s = 0, n = 0; for (let k = i - (win >> 1); k < i - (win >> 1) + win; k++) { if (k >= 0 && k < a.length) s += a[k]; n++; } return s / n; });
    zoneNames.forEach((name, zi) => {
      const z = scene.zones[name], c = box(Array.from(count[zi])), st = box(Array.from(still[zi]));
      let jam = c.map((cv, i) => cv >= z.min_count && st[i] / Math.max(cv, 1e-6) >= 0.7 && (!z.signalled || signal[i] !== RED));
      jam = fillGaps(jam, 3 * SAMPLE_FPS);
      jams[name] = clean(flagsToSegs(times, jam), 3, 10, dur);
    });
    ev.congestion = clean(Object.values(jams).flat(), 3, 10, dur);

    // stopped_vehicle
    segs = [];
    const visible = (v, i) => { const x1 = v.x[i] - v.w[i] / 2, y1 = v.y[i] - v.h[i], x2 = v.x[i] + v.w[i] / 2, y2 = v.y[i]; return x1 > 0.012 && y1 > 0.012 && x2 < 0.988 && y2 < 0.988; };
    for (const v of vehicles) {
      for (const [i, j] of runs(fillGaps(v.speed.map((s) => s < STILL), 3))) {
        if (v.t[j] - v.t[i] < 10) continue;
        if (!(onRoad(v.x[i], v.y[i]) && visible(v, i) && v.h[i] >= 0.035) || inParking(v.x[i], v.y[i])) continue;
        const zone = zoneOf(v.x[i], v.y[i]);
        if (zone === "near_approach") {
          const reds = Array.from({ length: 8 }, (_, q) => sigAt(v.t[i] + ((v.t[j] - v.t[i]) * q) / 7) === RED);
          if (reds.filter(Boolean).length / 8 > 0.3) continue;
        }
        const tm = (v.t[i] + v.t[j]) / 2, k = v.at(tm);
        if (k !== null && vehicles.some((o) => { if (o === v) return false; const m = o.at(tm); return m !== null && o.speed[m] <= STILL && Math.hypot(o.x[m] - v.x[k], o.y[m] - v.y[k]) < 3 * Math.max(v.w[k], v.h[k]); })) continue;
        if (zone && (jams[zone] || []).some(([s, e]) => s <= tm && tm <= e)) continue;
        segs.push([v.t[i], v.t[j] + DT]);
      }
    }
    ev.stopped_vehicle = clean(segs, 2, 10, dur);

    // road_obstacle (animals)
    segs = [];
    for (const a of tracks.filter((t) => ANIMALS.has(t.cls) && t.conf.reduce((p, q) => p + q, 0) / t.conf.length >= 0.35))
      for (const [i, j] of runs(fillGaps(a.x.map((x, k) => onRoad(x, a.y[k])), 3))) if (a.t[j] - a.t[i] >= 2) segs.push([a.t[i], a.t[j] + DT]);
    ev.road_obstacle = clean(segs, 3, 2, dur);

    const events = [];
    for (const [label, ss] of Object.entries(ev)) for (const [s, e] of merge(ss, 0)) events.push([s, e, label]);
    return events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }

  /* ---------------- run ---------------- */
  async function run(file, progress) {
    await init(progress);
    const video = document.createElement("video");
    video.muted = true; video.playsInline = true; video.preload = "auto";
    video.src = URL.createObjectURL(file);
    await new Promise((res, rej) => { video.onloadeddata = res; video.onerror = () => rej(new Error("this browser cannot decode the video (try an H.264 .mp4)")); });
    const dur = video.duration;
    if (!isFinite(dur) || dur <= 0) throw new Error("could not read the video duration");
    if (dur > MAX_SECONDS) throw new Error(`the video is ${Math.round(dur)} s long; the demo accepts up to ${MAX_SECONDS} s`);

    progress("aligning the view to our reference", 0.03);
    const shift = await align(video);
    const tracker = new Tracker(), risk = new Risk();
    const rows = [], times = [], lamps = [], riskCurve = [];
    const n = Math.floor(dur * SAMPLE_FPS);
    const started = performance.now();
    for (let k = 0; k < n; k++) {
      const t = +(k * DT).toFixed(3);
      await seek(video, t);
      const out = tracker.update(await detect(video), t);
      rows.push(...out); times.push(t); lamps.push(lampScores(video, shift));
      riskCurve.push([t, +risk.step(out, t).toFixed(4)]);
      if (k % 5 === 0) {
        const el = (performance.now() - started) / 1000, eta = (el / (k + 1)) * (n - k - 1);
        progress(`detecting & tracking: ${Math.round(t)} / ${Math.round(dur)} s of video, ~${Math.ceil(eta)} s left`, 0.05 + 0.9 * (k + 1) / n);
      }
    }
    progress("applying the event rules", 0.97);
    const signal = stateTimeline(lamps);
    const refRows = rows.map(([t, id, c, s, x1, y1, x2, y2]) => [t, id, c, s, x1 + shift.dx, y1 + shift.dy, x2 + shift.dx, y2 + shift.dy]);
    const events = rules(buildTracks(refRows), times, signal, dur);
    return {
      video: file.name, meta: { duration: +dur.toFixed(2), width: video.videoWidth, height: video.videoHeight },
      events, risk: riskCurve, signal: times.map((t, i) => [t, signal[i]]).filter((_, i) => i % 5 === 0),
      rows, shift, url: video.src,
    };
  }

  /* ---------------- annotated playback: overlay on the original video ---------------- */
  function attachOverlay(videoEl, canvas, res, colors) {
    const byT = new Map();
    for (const r of res.rows) { const key = r[0].toFixed(3); if (!byT.has(key)) byT.set(key, []); byT.get(key).push(r); }
    const sampleTimes = [...byT.keys()].map(Number).sort((a, b) => a - b);
    const toVid = (poly) => poly.map(([x, y]) => [x - res.shift.dx, y - res.shift.dy]);
    const draw = () => {
      const w = (canvas.width = videoEl.clientWidth * devicePixelRatio), h = (canvas.height = videoEl.clientHeight * devicePixelRatio);
      const ctx = canvas.getContext("2d"), t = videoEl.currentTime;
      ctx.clearRect(0, 0, w, h);
      ctx.lineWidth = Math.max(1, w / 900);
      ctx.strokeStyle = "rgba(255,255,255,.55)";
      for (const p of Object.values(scene.crosswalks)) { ctx.beginPath(); toVid(p).forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h))); ctx.closePath(); ctx.stroke(); }
      ctx.strokeStyle = "#ff3c3c"; ctx.beginPath(); toVid(scene.stop_line).forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h))); ctx.stroke();
      let lo = 0, hi = sampleTimes.length - 1, best = -1;
      while (lo <= hi) { const m = (lo + hi) >> 1; if (sampleTimes[m] <= t + 1e-6) { best = m; lo = m + 1; } else hi = m - 1; }
      if (best >= 0 && t - sampleTimes[best] <= 0.25) {
        ctx.font = `${Math.max(10, w / 110)}px system-ui`;
        for (const [, id, c, , x1, y1, x2, y2] of byT.get(sampleTimes[best].toFixed(3))) {
          const col = c === PERSON ? "#00c8ff" : VEHICLES.has(c) ? "#50dc50" : "#ffc800";
          ctx.strokeStyle = col; ctx.fillStyle = col;
          ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
          ctx.fillText(`${NAMES[c] || "?"} ${id}`, x1 * w, y1 * h - 3);
        }
      }
      let y0 = Math.max(18, h / 28);
      ctx.font = `bold ${Math.max(12, w / 70)}px system-ui`;
      for (const [s, e, lab] of res.events) {
        if (t < s || t >= e) continue;
        const txt = `${lab.replace(/_/g, " ").toUpperCase()}  ${s.toFixed(1)}-${e.toFixed(1)}s`, tw = ctx.measureText(txt).width;
        ctx.fillStyle = colors[lab] || "#888"; ctx.fillRect(8, y0 - h / 32, tw + 12, h / 26);
        ctx.fillStyle = "#fff"; ctx.fillText(txt, 14, y0);
        y0 += h / 22;
      }
      if (!videoEl.paused) requestAnimationFrame(draw);
    };
    ["play", "seeked", "loadeddata", "timeupdate"].forEach((e) => videoEl.addEventListener(e, () => requestAnimationFrame(draw)));
    window.addEventListener("resize", () => requestAnimationFrame(draw));
  }

  return { run, attachOverlay };
})();
