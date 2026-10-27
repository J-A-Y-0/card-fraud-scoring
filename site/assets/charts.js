// Small SVG chart kit shared by the report, the deck and the decision tool.
// Every chart reads pipeline-exported JSON from site/data; nothing is hand-entered.

const NS = "http://www.w3.org/2000/svg";

export const fmt = {
  usd: (v) => "$" + Math.round(v).toLocaleString("en-US"),
  usdK: (v) => Math.abs(v) >= 1e6 ? "$" + parseFloat((v / 1e6).toFixed(Math.abs(v) >= 1e7 ? 0 : 2)) + "M" : (Math.abs(v) >= 1000 ? "$" + (v / 1000).toFixed(Math.abs(v) >= 100000 ? 0 : 1).replace(/\.0$/, "") + "k" : "$" + Math.round(v)),
  pct: (v, d = 0) => (v * 100).toFixed(d) + "%",
  num: (v, d = 0) => v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }),
};

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function text(parent, x, y, str, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

function tooltip(host) {
  host.style.position = "relative";
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.setAttribute("role", "status");
  host.appendChild(tip);
  return {
    show(html, x, y) {
      tip.innerHTML = html;
      tip.classList.add("on");
      const w = tip.offsetWidth, hw = host.clientWidth;
      tip.style.left = Math.min(Math.max(0, x + 12), hw - w) + "px";
      tip.style.top = Math.max(0, y - tip.offsetHeight - 10) + "px";
    },
    hide() { tip.classList.remove("on"); },
  };
}

function dayLabel(start, d) {
  const t = new Date(start + "T00:00:00");
  t.setDate(t.getDate() + d);
  return t.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/**
 * Burst strip: one row per compromised card across the out-of-time window.
 * Bars are daily fraud dollars, red if lost and blue if stopped by the card block.
 */
export function burstStrip(host, data, { width = 680, rowH = 24, labelW = 62, valueW = 150, rows = null } = {}) {
  const cards = rows ? data.cards.slice(0, rows) : data.cards;
  const top = 26, bottom = 26;
  const plotW = width - labelW - valueW - 12;
  const height = top + cards.length * rowH + bottom;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", class: "chart", role: "img",
    "aria-label": `Daily fraud dollars for ${cards.length} compromised cards, lost versus stopped` }, host);
  const dayW = plotW / data.days;
  const x = (d) => labelW + d * dayW;
  const maxDay = Math.max(...cards.flatMap((c) => c.days.map((d) => d.lost + d.stopped)));
  const h = (v) => (v <= 0 ? 0 : Math.max(2, Math.sqrt(v / maxDay) * (rowH - 7)));
  const tip = tooltip(host);

  // Month ticks
  const g = el("g", { class: "axis" }, svg);
  for (const d of [0, 14, 30, 44, data.days - 1]) {
    el("line", { x1: x(d) + dayW / 2, x2: x(d) + dayW / 2, y1: top - 6, y2: height - bottom + 4 }, g);
    text(svg, x(d) + dayW / 2, height - 8, dayLabel(data.start, d), { "text-anchor": "middle" });
  }
  text(svg, labelW + plotW + 12, 12, "LOST", { "font-size": 10, "letter-spacing": "0.08em" });
  text(svg, width, 12, "STOPPED", { "font-size": 10, "letter-spacing": "0.08em", "text-anchor": "end" });

  cards.forEach((c, i) => {
    const base = top + (i + 1) * rowH - 4;
    const row = el("g", {}, svg);
    text(row, 0, base - 3, c.card, { class: "label-strong" });
    // Active span (first fraud to block) and blocked span after it
    const blockDay = c.first_alert_day === null ? null : c.first_alert_day + 1;
    el("line", { x1: x(c.first_fraud_day), x2: x(blockDay ?? data.days), y1: base, y2: base,
      stroke: "var(--rule-strong)", "stroke-width": 1 }, row);
    if (blockDay !== null && blockDay < data.days) {
      el("line", { x1: x(blockDay), x2: x(data.days), y1: base, y2: base,
        stroke: "var(--stopped)", "stroke-width": 1, "stroke-dasharray": "2 3", opacity: 0.7 }, row);
      el("rect", { x: x(blockDay) - 1, y: base - rowH + 8, width: 2, height: rowH - 8, rx: 1, fill: "var(--ink)" }, row);
    }
    for (const d of c.days) {
      const total = d.lost + d.stopped;
      if (total <= 0) continue;
      const bw = Math.max(1.5, dayW - 2);
      let y = base;
      for (const [k, v] of [["lost", d.lost], ["stopped", d.stopped]]) {
        if (v <= 0) continue;
        const hh = h(total) * (v / total);
        el("rect", { x: x(d.day) + 1, y: y - hh, width: bw, height: Math.max(1, hh), rx: 1,
          fill: `var(--${k})` }, row);
        y -= hh;
      }
      const hit = el("rect", { x: x(d.day), y: base - rowH + 4, width: dayW, height: rowH - 2, fill: "transparent" }, row);
      hit.addEventListener("mouseenter", (e) => {
        const r = host.getBoundingClientRect();
        tip.show(`<b>${c.card}</b> · ${dayLabel(data.start, d.day)}<br>` +
          (d.lost ? `lost ${fmt.usd(d.lost)}<br>` : "") + (d.stopped ? `stopped ${fmt.usd(d.stopped)}` : ""),
          e.clientX - r.left, e.clientY - r.top);
      });
      hit.addEventListener("mouseleave", () => tip.hide());
    }
    text(row, labelW + plotW + 12, base - 3, fmt.usdK(c.lost));
    text(row, width, base - 3, fmt.usdK(c.stopped), { "text-anchor": "end", class: "label-strong" });
  });
  return svg;
}

/** Dumbbell: what each split strategy's test set claimed vs what out-of-time delivered. */
export function dumbbell(host, rows, { width = 640, rowH = 64, domain = [0.5, 0.9], highlight = "random", left = 150, fs = 11 } = {}) {
  const right = 30, top = 30, bottom = 20 + 1.3 * fs;
  const height = top + rows.length * rowH + bottom;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", class: "chart", role: "img",
    "aria-label": "Test FDR versus out-of-time FDR for each split strategy" }, host);
  const x = (v) => left + ((v - domain[0]) / (domain[1] - domain[0])) * (width - left - right);
  const grid = el("g", { class: "grid" }, svg);
  for (let v = domain[0]; v <= domain[1] + 1e-9; v += 0.1) {
    el("line", { x1: x(v), x2: x(v), y1: top - 8, y2: height - bottom }, grid);
    text(svg, x(v), height - 8, fmt.pct(v), { "text-anchor": "middle", "font-size": fs });
  }
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    const hot = r.key === highlight;
    text(svg, 0, y - 0.35 * fs, r.label, { class: "label-strong", "font-size": fs * 1.1 });
    text(svg, 0, y + 1.15 * fs, r.note, { "font-size": fs * 0.95 });
    el("line", { x1: x(r.oot), x2: x(r.test), y1: y, y2: y, stroke: hot ? "var(--lost)" : "var(--rule-strong)",
      "stroke-width": hot ? 3 : 2, "stroke-linecap": "round" }, svg);
    el("circle", { cx: x(r.test), cy: y, r: 6, fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 2 }, svg);
    el("circle", { cx: x(r.oot), cy: y, r: 6.5, fill: "var(--ink)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
    text(svg, x(r.test) + 12, y + fs * 0.36, fmt.pct(r.test, 1), { class: "label-strong", "font-size": fs });
    text(svg, x(r.oot) - 12, y + fs * 0.36, fmt.pct(r.oot, 1), { "text-anchor": "end", "font-size": fs });
  });
  return svg;
}

/** Net value across alert rates for one policy, with the current choice marked. */
export function valueCurve(host, points, current, { width = 640, height = 300 } = {}) {
  host.querySelectorAll("svg").forEach((s) => s.remove());
  const left = 64, right = 16, top = 16, bottom = 36;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", class: "chart", role: "img",
    "aria-label": "Net value by alert rate" }, host);
  const xs = points.map((p) => p.rate), ys = points.map((p) => p.net);
  const xMax = Math.max(...xs);
  const yMin = Math.min(0, ...ys), yMax = Math.max(...ys) * 1.08;
  const x = (v) => left + (v / xMax) * (width - left - right);
  const y = (v) => top + (1 - (v - yMin) / (yMax - yMin)) * (height - top - bottom);
  const grid = el("g", { class: "grid" }, svg);
  const step = niceStep(yMax - yMin);
  for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
    el("line", { x1: left, x2: width - right, y1: y(v), y2: y(v) }, grid);
    text(svg, left - 8, y(v) + 4, fmt.usdK(v), { "text-anchor": "end" });
  }
  for (const v of [0, 0.02, 0.04, 0.06, 0.08, 0.1].filter((v) => v <= xMax + 1e-9)) {
    text(svg, x(v), height - 12, fmt.pct(v), { "text-anchor": "middle" });
  }
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.rate).toFixed(1)},${y(p.net).toFixed(1)}`).join("");
  el("path", { d, fill: "none", stroke: "var(--stopped)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  if (current) {
    el("line", { x1: x(current.rate), x2: x(current.rate), y1: top, y2: height - bottom,
      stroke: "var(--ink-3)", "stroke-dasharray": "3 3" }, svg);
    el("circle", { cx: x(current.rate), cy: y(current.net), r: 6, fill: "var(--stopped)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
  }
  const tip = tooltip(host);
  const cross = el("line", { y1: top, y2: height - bottom, stroke: "var(--rule-strong)", opacity: 0 }, svg);
  const overlay = el("rect", { x: left, y: top, width: width - left - right, height: height - top - bottom, fill: "transparent" }, svg);
  overlay.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const vx = ((e.clientX - r.left) / r.width) * width;
    const p = points.reduce((a, b) => (Math.abs(x(b.rate) - vx) < Math.abs(x(a.rate) - vx) ? b : a));
    cross.setAttribute("x1", x(p.rate)); cross.setAttribute("x2", x(p.rate)); cross.setAttribute("opacity", 1);
    const hr = host.getBoundingClientRect();
    tip.show(`alert rate ${fmt.pct(p.rate, 2)}<br>net value <b>${fmt.usd(p.net)}</b><br>${fmt.num(p.alerts)} alerts`,
      e.clientX - hr.left, e.clientY - hr.top);
  });
  overlay.addEventListener("mouseleave", () => { tip.hide(); cross.setAttribute("opacity", 0); });
  return svg;
}

function niceStep(span) {
  const raw = span / 4, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
}

function hoverTarget(svg, host, tip, node, html) {
  node.addEventListener("mouseenter", (e) => {
    const r = host.getBoundingClientRect();
    tip.show(html, e.clientX - r.left, e.clientY - r.top);
  });
  node.addEventListener("mouseleave", () => tip.hide());
}

/** Vertical bars for one measure over categories. */
export function bars(host, items, { width = 320, height = 180, yFmt = fmt.num, color = "var(--ink-2)",
  labelEvery = 1, valueLabels = false, fs = 10.5, title = "" } = {}) {
  const left = 44, right = 6, top = 14, bottom = 22;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", class: "chart", role: "img", "aria-label": title }, host);
  const tip = tooltip(host);
  const yMax = Math.max(...items.map((d) => d.value)) * 1.1;
  const step = niceStep(yMax);
  const y = (v) => top + (1 - v / yMax) * (height - top - bottom);
  const bw = (width - left - right) / items.length;
  const g = el("g", { class: "grid" }, svg);
  for (let v = 0; v <= yMax; v += step) {
    el("line", { x1: left, x2: width - right, y1: y(v), y2: y(v) }, g);
    text(svg, left - 6, y(v) + 3.5, yFmt(v), { "text-anchor": "end", "font-size": fs });
  }
  items.forEach((d, i) => {
    const x = left + i * bw;
    const h = Math.max(0, y(0) - y(d.value));
    const w = items.length <= 4 ? bw * 0.5 : Math.max(1, bw - 2);  // few categories: separate, narrower bars
    const r = el("rect", { x: x + (bw - w) / 2, y: y(d.value), width: w, height: h, rx: Math.min(3, bw / 4),
      fill: d.color || color }, svg);
    hoverTarget(svg, host, tip, r, `${d.label}<br><b>${yFmt(d.value)}</b>`);
    if (i % labelEvery === 0) text(svg, x + bw / 2, height - 6, d.label, { "text-anchor": "middle", "font-size": fs });
    if (valueLabels) text(svg, x + bw / 2, y(d.value) - 4, yFmt(d.value), { "text-anchor": "middle", "font-size": fs, class: "label-strong" });
  });
  return svg;
}

/** Lines sharing one y scale. Series: [{name, color, dash, points:[{x,y}]}]. Direct-labeled at the right end. */
export function lines(host, series, { width = 320, height = 200, xFmt = fmt.num, yFmt = fmt.num, xTicks = null,
  yDomain = null, logX = false, labelW = 0, fs = 10.5, title = "", markers = false } = {}) {
  const left = 46, right = 8 + labelW, top = 12, bottom = 24;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", class: "chart", role: "img", "aria-label": title }, host);
  const tip = tooltip(host);
  const all = series.flatMap((s) => s.points);
  const tx = (v) => (logX ? Math.log10(v) : v);
  const xMin = Math.min(...all.map((p) => tx(p.x))), xMax = Math.max(...all.map((p) => tx(p.x)));
  const [yMin, yMax] = yDomain || [Math.min(0, ...all.map((p) => p.y)), Math.max(...all.map((p) => p.y)) * 1.08];
  const x = (v) => left + ((tx(v) - xMin) / (xMax - xMin || 1)) * (width - left - right);
  const y = (v) => top + (1 - (v - yMin) / (yMax - yMin)) * (height - top - bottom);
  const g = el("g", { class: "grid" }, svg);
  const step = niceStep(yMax - yMin);
  for (let v = Math.ceil(yMin / step) * step; v <= yMax + 1e-9; v += step) {
    el("line", { x1: left, x2: width - right, y1: y(v), y2: y(v) }, g);
    text(svg, left - 6, y(v) + 3.5, yFmt(v), { "text-anchor": "end", "font-size": fs });
  }
  for (const v of xTicks || [...new Set(all.map((p) => p.x))]) {
    text(svg, x(v), height - 7, xFmt(v), { "text-anchor": "middle", "font-size": fs });
  }
  for (const s of series) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join("");
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "none",
      "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    if (markers) for (const p of s.points) el("circle", { cx: x(p.x), cy: y(p.y), r: 3, fill: s.color, stroke: "var(--paper)", "stroke-width": 1.5 }, svg);
    if (labelW) {
      const last = s.points.at(-1);
      text(svg, x(last.x) + 6, y(last.y) + 3.5, s.name, { "font-size": fs, class: "label-strong" });
    }
  }
  const cross = el("line", { y1: top, y2: height - bottom, stroke: "var(--rule-strong)", opacity: 0 }, svg);
  const overlay = el("rect", { x: left, y: top, width: width - left - right, height: height - top - bottom, fill: "transparent" }, svg);
  overlay.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const vx = ((e.clientX - r.left) / r.width) * width;
    const xs = [...new Set(all.map((p) => p.x))];
    const px = xs.reduce((a, b) => (Math.abs(x(b) - vx) < Math.abs(x(a) - vx) ? b : a));
    cross.setAttribute("x1", x(px)); cross.setAttribute("x2", x(px)); cross.setAttribute("opacity", 1);
    const rows = series.map((s) => { const p = s.points.find((q) => q.x === px); return p ? `${s.name}: <b>${yFmt(p.y)}</b>` : ""; }).filter(Boolean);
    const hr = host.getBoundingClientRect();
    tip.show(`${xFmt(px)}<br>${rows.join("<br>")}`, e.clientX - hr.left, e.clientY - hr.top);
  });
  overlay.addEventListener("mouseleave", () => { tip.hide(); cross.setAttribute("opacity", 0); });
  return svg;
}

/** Tornado: outcome when each input moves to the low / high end of its range. */
export function tornado(host, rows, base, { width = 640, rowH = 30, labelW = 220, valFmt = fmt.usdK, fs = 11, title = "" } = {}) {
  const right = 56, top = 22, bottom = 22;
  const height = top + rows.length * rowH + bottom;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", class: "chart", role: "img", "aria-label": title }, host);
  const vals = rows.flatMap((r) => [r.atLow, r.atHigh]).concat(base);
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.04, valW = 52;
  const x = (v) => labelW + valW + ((v - lo + pad) / (hi - lo + 2 * pad)) * (width - labelW - valW - right);
  el("line", { x1: x(base), x2: x(base), y1: top - 10, y2: height - bottom + 4, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
  text(svg, x(base), top - 12, `default ${valFmt(base)}`, { "text-anchor": "middle", "font-size": fs, class: "label-strong" });
  rows.forEach((r, i) => {
    const yy = top + i * rowH + 6;
    text(svg, 0, yy + rowH / 2 - 2, r.label, { "font-size": fs });
    const a = Math.min(r.atLow, r.atHigh), b = Math.max(r.atLow, r.atHigh);
    el("rect", { x: x(a), y: yy, width: Math.max(2, x(b) - x(a)), height: rowH - 12, rx: 3, fill: "var(--stopped)" }, svg);
    text(svg, x(b) + 6, yy + rowH / 2 - 2, `${valFmt(b)}`, { "font-size": fs - 1 });
    if (x(b) - x(a) > 2) text(svg, x(a) - 6, yy + rowH / 2 - 2, `${valFmt(a)}`, { "font-size": fs - 1, "text-anchor": "end" });
  });
  return svg;
}
