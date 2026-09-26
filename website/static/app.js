/* CrossWatch website: renders data/*.json produced by tools/eda.py and tools/export_results.py,
   and drives the live demo against website/server.py. No build step. */
"use strict";

const CLASS_COLORS = {
  accident: "#e62828", near_miss: "#ff8c00", red_light: "#ff3c3c", wrong_way: "#c800c8",
  illegal_u_turn: "#dc50b4", stopped_vehicle: "#ffc800", jaywalking: "#00aaff",
  failure_to_yield: "#5a5aff", illegal_turn: "#ff64a0", solid_line_crossing: "#dcdc64",
  stop_line: "#ffa050", congestion: "#b45050", road_obstacle: "#a0ff00", fire_smoke: "#b40000",
};
const OBJ_COLORS = { car: "#50dc50", person: "#00c8ff", bus: "#ff78dc", truck: "#b478ff", motorcycle: "#ffc800" };
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
};
const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const pretty = (lab) => lab.replace(/_/g, " ");
const getJSON = (u) => fetch(u).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${u}: ${r.status}`))));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/* ---------------- theme ---------------- */
(function theme() {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* storage blocked */ }
  if (saved) root.dataset.theme = saved;
  $("#themeBtn").addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ignore */ }
    location.reload();
  });
})();

function chartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--line");
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.maintainAspectRatio = false;
}

/* ---------------- timeline (SVG, clickable) ---------------- */
function renderTimeline(box, events, duration, onSeek) {
  box.innerHTML = "";
  const labels = [...new Set(events.map((e) => e[2]))].sort();
  const W = 1000, rowH = 22, top = 8, left = 130, axis = 22;
  const H = top + Math.max(1, labels.length) * rowH + axis;
  const x = (t) => left + (t / duration) * (W - left - 10);
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const mk = (tag, a) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); return n; };
  if (!labels.length) {
    const t = mk("text", { x: W / 2, y: 22, "text-anchor": "middle", fill: css("--muted"), "font-size": 13 });
    t.textContent = "no events detected"; svg.append(t);
  }
  labels.forEach((lab, i) => {
    const y = top + i * rowH;
    svg.append(mk("rect", { x: left, y: y + 2, width: W - left - 10, height: rowH - 4, fill: css("--panel-2"), rx: 3 }));
    const t = mk("text", { x: left - 8, y: y + rowH / 2 + 4, "text-anchor": "end", fill: css("--text"), "font-size": 12 });
    t.textContent = pretty(lab); svg.append(t);
  });
  for (const [s, e, lab] of events) {
    const i = labels.indexOf(lab), y = top + i * rowH;
    const r = mk("rect", { x: x(s), y: y + 3, width: Math.max(3, x(e) - x(s)), height: rowH - 6, fill: CLASS_COLORS[lab] || "#888", rx: 3, class: "ev" });
    const title = mk("title", {}); title.textContent = `${pretty(lab)}: ${s.toFixed(1)}-${e.toFixed(1)} s`; r.append(title);
    r.addEventListener("click", () => onSeek && onSeek(s));
    svg.append(r);
  }
  const yA = top + Math.max(1, labels.length) * rowH;
  const step = duration > 240 ? 60 : duration > 60 ? 30 : 10;
  for (let t = 0; t <= duration; t += step) {
    svg.append(mk("line", { x1: x(t), x2: x(t), y1: top, y2: yA + 4, stroke: css("--line") }));
    const tx = mk("text", { x: x(t), y: yA + 16, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 });
    tx.textContent = fmt(t).replace(/\.\d$/, ""); svg.append(tx);
  }
  const cursor = mk("line", { x1: left, x2: left, y1: top, y2: yA, stroke: css("--accent"), "stroke-width": 2 });
  svg.append(cursor);
  svg.addEventListener("click", (ev) => {
    if (ev.target.classList.contains("ev")) return;
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    if (p.x >= left) onSeek && onSeek(((p.x - left) / (W - left - 10)) * duration);
  });
  box.append(svg);
  return (t) => { cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t)); };
}

function renderEventTable(table, events, onSeek) {
  table.innerHTML = "<tr><th>class</th><th>start</th><th>end</th><th>length</th></tr>";
  if (!events.length) table.append(el("tr", {}, el("td", { colspan: 4, class: "muted" }, "no events")));
  for (const [s, e, lab] of events) {
    table.append(el("tr", { onclick: () => onSeek(s) },
      el("td", {}, el("span", { class: "tag", style: `background:${CLASS_COLORS[lab] || "#888"}` }, pretty(lab))),
      el("td", {}, fmt(s)), el("td", {}, fmt(e)), el("td", {}, `${(e - s).toFixed(1)} s`)));
  }
}

const charts = {};
function riskChart(canvas, res) {
  if (!window.Chart) return;
  charts[canvas.id]?.destroy();
  const risk = res.risk || [];
  const sig = res.signal || [];
  const dur = res.meta.duration;
  const alarmPlugin = {
    id: "bands",
    beforeDatasetsDraw(c) {
      const { ctx, chartArea: a, scales: { x } } = c;
      ctx.save();
      for (let i = 0; i < sig.length; i++) {
        const t0 = sig[i][0], t1 = i + 1 < sig.length ? sig[i + 1][0] : dur;
        ctx.fillStyle = sig[i][1] === 1 ? "rgba(255,90,95,.10)" : sig[i][1] === 0 ? "rgba(53,201,122,.08)" : "transparent";
        ctx.fillRect(x.getPixelForValue(t0), a.top, x.getPixelForValue(t1) - x.getPixelForValue(t0), a.bottom - a.top);
      }
      for (const [s, e, lab] of res.events || []) {
        if (lab !== "accident" && lab !== "near_miss") continue;
        ctx.fillStyle = "rgba(230,40,40,.25)";
        ctx.fillRect(x.getPixelForValue(s), a.top, x.getPixelForValue(e) - x.getPixelForValue(s), a.bottom - a.top);
      }
      ctx.restore();
    },
  };
  charts[canvas.id] = new Chart(canvas, {
    type: "line",
    data: { datasets: [
      { label: "P(accident within 5 s)", data: risk.map(([t, s]) => ({ x: t, y: s })), borderColor: css("--accent"), borderWidth: 1.5, pointRadius: 0, fill: false },
      { label: "alarm threshold 0.5", data: [{ x: 0, y: 0.5 }, { x: dur, y: 0.5 }], borderColor: css("--red"), borderDash: [5, 4], borderWidth: 1, pointRadius: 0 },
    ] },
    options: {
      animation: false, parsing: false,
      interaction: { mode: "nearest", intersect: false, axis: "x" },
      scales: { x: { type: "linear", min: 0, max: dur, title: { display: true, text: "seconds (background: red / green phase of the approach signal)" } },
                y: { min: 0, max: 1 } },
      plugins: { legend: { labels: { boxWidth: 12 } } },
    },
    plugins: [alarmPlugin],
  });
}

/* ---------------- sections ---------------- */
async function loadTeam() {
  const team = await getJSON("data/team.json").catch(() => null);
  const grid = $("#teamGrid");
  if (!team) { grid.textContent = "team.json missing"; return; }
  for (const m of team.members) {
    const initials = m.name.split(/\s+/).map((p) => p[0]).join("").slice(0, 2).toUpperCase();
    grid.append(el("div", { class: "card member" },
      el("div", { class: "avatar" }, initials),
      el("h3", { style: "margin:4px 0 0" }, m.name),
      el("div", { class: "role" }, m.role),
      el("div", {}, el("b", {}, "Did: "), m.did),
      m.projects?.length ? el("div", {}, el("b", {}, "Proud of: "), el("ul", {}, ...m.projects.map((p) => el("li", {}, p)))) : null,
      el("div", { class: "links" }, ...Object.entries(m.links || {}).filter(([, u]) => u).map(([k, u]) => el("a", { href: u, target: "_blank", rel: "noopener" }, k)))));
  }
}

function loadRules() {
  const rows = [
    ["jaywalking", "rule", "A reliably detected person (≥ 3.5 % of frame height, conf ≥ 0.45) on the carriageway mask, off the islands and outside the crossings' walking band, for 1.5-60 s, moving ≥ 1 body height at walking speed (riders and tracker ID jumps excluded)."],
    ["failure_to_yield", "rule", "Moving vehicle inside a crossing while a walking pedestrian is on its painted stripes over the carriageway, within ~1.5 lane widths of the vehicle."],
    ["red_light", "rule", "Ground point clearly passes the stop line while the lamp phase has been red ≥ 4 s and stays red ≥ 1 s more (the approach's own lamp lags the heads we see); ends when the vehicle leaves the junction (≤ 8 s)."],
    ["stop_line", "rule", "Vehicle comes to a stop on red between the stop line and the far edge of the crossing and holds ≥ 2 s; ends at green. Cars stuck there since green count as congestion instead."],
    ["stopped_vehicle", "rule", "Vehicle still (< 0.12 heights/s) ≥ 10 s on the carriageway and fully in view; not in the kerb parking lane, not queued at the red, no stationary neighbour, not inside a jam."],
    ["wrong_way", "rule", "Heading opposite (≥ 135°) to a lane's dominant flow (≥ 60 % of learned headings in that cell), for ≥ 3 s over ≥ 5 % of the frame. Junction cells have no dominant flow, so turns never fire."],
    ["congestion", "rule", "A direction's zone holds ≥ 8 vehicles (5 on the far carriageway), ≥ 70 % still, for ≥ 10 s; on the signalled approach only during green."],
    ["road_obstacle", "rule", "Animal (COCO) on the carriageway ≥ 2 s. Debris is not detectable with COCO classes."],
    ["accident / near_miss", "off", "Not emitted in Part A: no reliable signal without labels, and a false class costs a whole macro-F1 slot. Covered by Part B's risk score instead."],
    ["illegal_u_turn / illegal_turn / solid_line_crossing / fire_smoke", "off", "Not emitted: the legality depends on signage we cannot verify from the samples; none were observed."],
  ];
  const t = $("#rulesTable");
  t.innerHTML = "<tr><th>class</th><th>type</th><th>logic</th></tr>";
  for (const [c, k, d] of rows) t.append(el("tr", {}, el("td", {}, el("code", {}, c)),
    el("td", {}, el("span", { class: `tag ${k === "rule" ? "rule" : k === "off" ? "off" : "learned"}` }, k === "off" ? "not emitted" : "rule-based")), el("td", {}, d)));
}

async function loadEDA() {
  const eda = await getJSON("data/eda.json").catch(() => null);
  if (!eda) return;
  const vids = eda.videos;
  const t = $("#edaTable");
  t.innerHTML = "<tr><th>video</th><th>resolution</th><th>fps</th><th>duration</th><th>size</th><th>cars / people / buses / trucks tracked</th><th>red / green (mean s)</th></tr>";
  for (const v of vids) {
    const u = v.unique_tracks;
    t.append(el("tr", {}, el("td", {}, v.name), el("td", {}, `${v.width}×${v.height}`), el("td", {}, v.fps),
      el("td", {}, `${v.duration} s`), el("td", {}, `${v.size_gb} GB`),
      el("td", {}, `${u.car} / ${u.person} / ${u.bus} / ${u.truck}`),
      el("td", {}, `${v.signal.mean_red_s ?? "-"} / ${v.signal.mean_green_s ?? "-"}`)));
  }
  const kp = $("#kpis");
  const tot = vids.reduce((a, v) => a + v.duration, 0);
  const tracks = vids.reduce((a, v) => a + Object.values(v.unique_tracks).reduce((x, y) => x + y, 0), 0);
  const kpi = (v, l) => el("div", { class: "kpi" }, el("div", { class: "v" }, v), el("div", { class: "l" }, l));
  kp.append(kpi(vids.length, "sample videos"), kpi(`${(tot / 60).toFixed(1)} min`, "of 4K footage"),
    kpi(tracks.toLocaleString(), "tracked road users"));
  window.__kpiBox = kp;

  // counts chart with tabs
  const tabs = $("#countTabs");
  const draw = (v) => {
    if (!window.Chart) return;
    charts.count?.destroy();
    const ds = Object.entries(v.counts_per_sec).map(([k, arr]) => ({
      label: k, data: arr.map((y, i) => ({ x: i, y })), borderColor: OBJ_COLORS[k], pointRadius: 0, borderWidth: 1.3, tension: 0.3 }));
    charts.count = new Chart($("#countChart"), { type: "line", data: { datasets: ds },
      options: { animation: false, parsing: false, scales: { x: { type: "linear", title: { display: true, text: "seconds" } }, y: { title: { display: true, text: "objects in view" } } },
        plugins: { legend: { labels: { boxWidth: 12 } } } } });
    const phases = v.signal.phases;
    $("#phaseBars").innerHTML = "";
    const bar = el("div", { style: "display:flex;height:22px;border-radius:6px;overflow:hidden" });
    for (const [st, s, e] of phases) bar.append(el("div", { title: `${st === 1 ? "red" : st === 0 ? "green" : "unknown"} ${s}-${e} s`,
      style: `flex:${Math.max(0.1, e - s)};background:${st === 1 ? "var(--red)" : st === 0 ? "var(--green)" : "#777"}` }));
    $("#phaseBars").append(el("div", { class: "muted", style: "margin-bottom:6px" }, v.name), bar);
    $("#phaseNote").textContent = `Mean red ${v.signal.mean_red_s ?? "-"} s, mean green ${v.signal.mean_green_s ?? "-"} s (complete phases only). The phase explains the near approach's discharge: in the samples ~95 % of stop-line crossings happen on green.`;
    $("#countNote").textContent = "The car curve saw-tooths with the ~75 s signal cycle: the queue builds on red and discharges on green.";
  };
  vids.forEach((v, i) => {
    const b = el("button", { class: i === 0 ? "on" : "", onclick: () => { tabs.querySelectorAll("button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); draw(v); } }, v.name);
    tabs.append(b);
  });
  if (vids.length) draw(vids[0]);

  if (window.Chart) {
    charts.light = new Chart($("#lightChart"), { type: "line", data: { datasets: vids.map((v, i) => ({
      label: v.name, data: v.light.map(([x, y]) => ({ x, y })), pointRadius: 0, borderWidth: 1.5,
      borderColor: ["#ffb020", "#3fb6ff", "#35c97a", "#ff5a5f"][i % 4] })) },
      options: { animation: false, parsing: false, scales: { x: { type: "linear", title: { display: true, text: "seconds" } }, y: { min: 0, max: 255 } } } });
  }
  const findings = eda.findings || [];
  $("#findings").append(...findings.map((f) => el("li", { html: f })));
}

async function loadResults() {
  const index = await getJSON("data/results.json").catch(() => []);
  const tabs = $("#resTabs"), video = $("#resVideo");
  const all = [];
  let setCursor = null;
  video.addEventListener("timeupdate", () => setCursor && setCursor(video.currentTime));
  const seek = (t) => { video.currentTime = Math.max(0, t - 1); video.play().catch(() => {}); video.scrollIntoView({ behavior: "smooth", block: "center" }); };
  const show = async (item, btn) => {
    tabs.querySelectorAll("button").forEach((x) => x.classList.remove("on")); btn.classList.add("on");
    const res = await getJSON(`data/results/${item.stem}.json`);
    video.src = `media/results/${item.stem}.mp4`;
    setCursor = renderTimeline($("#resTimeline"), res.events, res.meta.duration, seek);
    renderEventTable($("#resEvents"), res.events, seek);
    riskChart($("#resRisk"), res);
  };
  for (const [i, item] of index.entries()) {
    const b = el("button", { onclick: () => show(item, b) }, `${item.name} · ${item.n_events} events`);
    tabs.append(b);
    if (i === 0) show(item, b);
    all.push(getJSON(`data/results/${item.stem}.json`));
  }
  const results = await Promise.all(all);
  dashboard(results);
  const failures = await getJSON("data/failures.json").catch(() => []);
  $("#failures").append(...failures.map((f) => el("li", { html: f })));
}

function dashboard(results) {
  if (!window.Chart || !results.length) return;
  const byClass = {};
  let minutes = 0;
  for (const r of results) { minutes += r.meta.duration / 60; for (const [, , lab] of r.events) byClass[lab] = (byClass[lab] || 0) + 1; }
  const labs = Object.keys(byClass).sort((a, b) => byClass[b] - byClass[a]);
  charts.dc = new Chart($("#dashClass"), { type: "bar", data: { labels: labs.map(pretty),
    datasets: [{ label: "events", data: labs.map((l) => byClass[l]), backgroundColor: labs.map((l) => CLASS_COLORS[l]) }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } } } });
  charts.dr = new Chart($("#dashRate"), { type: "bar", data: { labels: results.map((r) => r.video),
    datasets: labs.map((l) => ({ label: pretty(l), backgroundColor: CLASS_COLORS[l],
      data: results.map((r) => r.events.filter((e) => e[2] === l).length / (r.meta.duration / 60)) })) },
    options: { scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: "events / minute" } } },
      plugins: { legend: { labels: { boxWidth: 10 } } } } });
  const total = Object.values(byClass).reduce((a, b) => a + b, 0);
  window.__kpiBox?.append(el("div", { class: "kpi" }, el("div", { class: "v" }, total), el("div", { class: "l" }, "events found in samples")),
    el("div", { class: "kpi" }, el("div", { class: "v" }, (total / minutes).toFixed(1)), el("div", { class: "l" }, "events per minute")));
}

async function loadReport() {
  const r = await getJSON("data/report.json").catch(() => null);
  if (!r) return;
  const box = $("#reportBody");
  for (const sec of r.sections) {
    box.append(el("h3", {}, sec.title));
    if (sec.html) box.append(el("div", { html: sec.html }));
    if (sec.items) box.append(el("ul", {}, ...sec.items.map((i) => el("li", { html: i }))));
  }
  const links = $("#linkList");
  for (const [k, u] of Object.entries(r.links || {})) links.append(el("li", {}, el("b", {}, `${k}: `), el("a", { href: u, target: "_blank", rel: "noopener" }, u)));
}

/* ---------------- live demo ---------------- */
function demo() {
  const input = $("#file"), drop = $("#drop"), run = $("#runBtn");
  const bar = $("#demoBar"), stage = $("#demoStage"), err = $("#demoErr");
  let file = null;
  const pick = (f) => {
    err.textContent = "";
    if (!f) return;
    if (!/\.mp4$/i.test(f.name)) { err.textContent = "Please choose an .mp4 file."; return; }
    if (f.size > 200 * 1024 * 1024) { err.textContent = "File is larger than 200 MB."; return; }
    file = f; $("#fileName").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`; run.disabled = false;
  };
  input.addEventListener("change", () => pick(input.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); });

  run.addEventListener("click", () => {
    if (!file) return;
    run.disabled = true; err.textContent = ""; $("#demoOut").hidden = true;
    const fd = new FormData(); fd.append("video", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "api/jobs");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { bar.style.width = `${(e.loaded / e.total) * 20}%`; stage.textContent = `uploading ${Math.round((e.loaded / e.total) * 100)}%`; } };
    xhr.onload = () => {
      if (xhr.status !== 200) { let m = xhr.responseText; try { m = JSON.parse(m).detail; } catch (e) { /* raw */ } err.textContent = m || `upload failed (${xhr.status})`; run.disabled = false; return; }
      poll(JSON.parse(xhr.responseText).id);
    };
    xhr.onerror = () => { err.textContent = "Upload failed: is the demo server reachable?"; run.disabled = false; };
    xhr.send(fd);
  });

  const poll = async (id) => {
    try {
      const j = await getJSON(`api/jobs/${id}`);
      if (j.state === "error") { err.textContent = j.error; run.disabled = false; return; }
      bar.style.width = `${20 + 80 * (j.progress || 0)}%`;
      stage.textContent = j.state === "queued" ? `queued (position ${j.queue_position})` : `${j.stage} · ${Math.round((j.progress || 0) * 100)}%`;
      if (j.state === "done") { showDemo(id, j.result); run.disabled = false; return; }
    } catch (e) { stage.textContent = "waiting for server..."; }
    setTimeout(() => poll(id), 1000);
  };

  const showDemo = (id, res) => {
    $("#demoOut").hidden = false;
    stage.textContent = `done: ${res.events.length} events`;
    const v = $("#demoVideo");
    v.src = `api/jobs/${id}/video`;
    const seek = (t) => { v.currentTime = Math.max(0, t - 1); v.play().catch(() => {}); };
    const setCursor = renderTimeline($("#demoTimeline"), res.events, res.meta.duration, seek);
    v.ontimeupdate = () => setCursor(v.currentTime);
    renderEventTable($("#demoEvents"), res.events, seek);
    riskChart($("#demoRisk"), res);
    $("#demoJson").href = URL.createObjectURL(new Blob([JSON.stringify({ video: res.video, events: res.events }, null, 1)], { type: "application/json" }));
  };
}

/* nav highlighting */
function navSpy() {
  const links = [...document.querySelectorAll("nav.top .links a")];
  const obs = new IntersectionObserver((ents) => {
    for (const e of ents) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
  }, { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("section[id]").forEach((s) => obs.observe(s));
}

window.addEventListener("DOMContentLoaded", async () => {
  chartDefaults();
  loadRules(); demo(); navSpy();
  await Promise.all([loadTeam(), loadEDA(), loadReport()]);
  await loadResults();
});
