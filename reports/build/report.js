/* What every report's charts share: SVG helpers, tooltips, resizing, tables.
   Inlined ahead of each page's own script, which reads `ReportKit`. */
const ReportKit = (() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const int = n => Math.round(n).toLocaleString('en-US');
  const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  function label(parent, x, y, str, cls, attrs) {
    const t = svg('text', Object.assign({ x, y, class: cls }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }
  function plot(box, W, H, title) {
    return svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'plot', role: 'img', 'aria-label': title || '' }, box);
  }

  const measureCtx = document.createElement('canvas').getContext('2d');
  function measure(str, size) {
    measureCtx.font = `${size || 12}px Archivo, system-ui, sans-serif`;
    return measureCtx.measureText(str).width;
  }

  function niceTicks(max, count) {
    const raw = Math.max(max, 1e-9) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
    const out = [];
    for (let i = 0; i * step <= max + step * 0.999; i++) out.push(+(i * step).toPrecision(10));
    return out;
  }

  function barRight(x, y, w, h) {
    if (w <= 0) return '';
    const r = Math.min(4, w, h / 2);
    return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`;
  }
  function barUp(x, y, w, h) {
    if (w <= 0 || h <= 0) return '';
    const r = Math.min(4, w / 2, h);
    return `M${x},${y + h}v${-(h - r)}a${r},${r} 0 0 1 ${r},${-r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`;
  }

  /* ---------- tooltip ---------- */
  const tip = document.getElementById('tip');
  function showTip(evt, node, content) {
    tip.replaceChildren();
    if (content.title) {
      const head = document.createElement('div');
      head.className = 'tip-title';
      head.textContent = content.title;
      tip.append(head);
    }
    for (const [value, what] of content.lines) {
      const row = document.createElement('div');
      row.className = 'tip-row';
      const v = document.createElement('strong');
      v.textContent = value;
      const w = document.createElement('span');
      w.textContent = what;
      row.append(v, w);
      tip.append(row);
    }
    tip.hidden = false;
    let x, y;
    if (evt.type === 'pointermove') { x = evt.clientX; y = evt.clientY; }
    else { const r = node.getBoundingClientRect(); x = r.left + r.width / 2; y = r.top; }
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(Math.max(8, x + 14), window.innerWidth - w - 8)}px`;
    tip.style.top = `${y - h - 12 < 8 ? y + 18 : y - h - 12}px`;
  }
  function hover(node, content, mark, focusable) {
    node.classList.add('hit');
    if (focusable) {
      node.setAttribute('tabindex', '0');
      const words = content.lines.map(([v, w]) => `${v} ${w}`);
      node.setAttribute('aria-label', [content.title, ...words].filter(Boolean).join(', '));
    }
    const marks = mark ? [].concat(mark) : [];
    const on = e => { showTip(e, node, content); marks.forEach(m => m.classList.add('is-active')); };
    const off = () => { tip.hidden = true; marks.forEach(m => m.classList.remove('is-active')); };
    node.addEventListener('pointermove', on);
    node.addEventListener('pointerleave', off);
    node.addEventListener('focus', on);
    node.addEventListener('blur', off);
  }
  window.addEventListener('scroll', () => { tip.hidden = true; }, { passive: true });

  /* ---------- charts redraw at their container's width ---------- */
  const renders = [];
  function mount(id, draw) {
    const box = document.getElementById(id);
    if (!box) return;
    let width = -1;
    const render = () => {
      width = Math.round(box.clientWidth);
      box.replaceChildren();
      if (width > 0) draw(box, width);
    };
    renders.push(render);
    new ResizeObserver(() => { if (Math.round(box.clientWidth) !== width) render(); }).observe(box);
  }
  // Text is measured to lay charts out; measure again once the web fonts arrive.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => renders.forEach(r => r()));

  /* ---------- tables ---------- */
  function fillTable(id, columns, rows) {
    const table = document.getElementById(id);
    if (!table) return;
    const head = document.createElement('thead');
    const tr = document.createElement('tr');
    for (const c of columns) {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = c.label;
      if (c.num) th.className = 'num';
      tr.append(th);
    }
    head.append(tr);
    const body = document.createElement('tbody');
    for (const r of rows) {
      const row = document.createElement('tr');
      if (r.rowClass) row.className = r.rowClass;
      for (const c of columns) {
        const td = document.createElement('td');
        const v = c.value(r);
        if (v instanceof Node) td.append(v); else td.textContent = v;
        const cls = [c.num ? 'num' : '', c.cls ? c.cls(r) : ''].filter(Boolean).join(' ');
        if (cls) td.className = cls;
        row.append(td);
      }
      body.append(row);
    }
    table.replaceChildren(head, body);
  }
  function codeNode(text) { const c = document.createElement('code'); c.textContent = text; return c; }
  function chip(role) { const s = document.createElement('span'); s.className = `chip ${role}`; s.textContent = role; return s; }
  function stacked(main, sub) {
    const frag = document.createDocumentFragment();
    frag.append(main);
    const s = document.createElement('span');
    s.className = 'sub';
    s.textContent = sub;
    frag.append(s);
    return frag;
  }

  return {
    data: JSON.parse(document.getElementById('report-data').textContent),
    int, pct, svg, label, plot, measure, niceTicks, barRight, barUp,
    hover, mount, fillTable, codeNode, chip, stacked,
  };
})();
