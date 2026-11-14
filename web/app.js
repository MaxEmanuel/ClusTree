'use strict';
// Front end of the demonstration. All computation happens in the C++ server; this file
// renders the points, the bounding rectangles of the index nodes and the query results.

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

const S = {
  state: null,        // /api/state
  xy: null,           // Float32Array, 2 per point (projected)
  truth: null,        // Int32Array or null
  partition: null,    // Int32Array: node of the selected tree level per point
  runLabels: null,    // Int32Array: labels of the selected run
  selectedRun: null,  // run id
  ari: {},            // run id -> ARI vs. selected run
  tree: [],           // [depth, leaf, x0, y0, x1, y1, fanout] down to the selected depth
  depth: 0,
  colorIdx: null,     // Uint8Array: palette slot per point (8 = noise, 9 = unlabelled)
  view: { cx: 0.5, cy: 0.5, scale: 1 },
  tool: 'pan',
  query: null,        // last query: shape, result ids, visited nodes
  drag: null,
  busy: 0,
};

const METHOD_NAMES = {
  'index-partition': 'Tree level (no algorithm)',
  'kmeans-tree': 'k-means on tree nodes',
  'kmeans-tree-seeded': 'k-means: tree nodes, then points',
  'kmeans-baseline': 'k-means, conventional',
  'dbscan-tree-dense': 'DBSCAN, dense subtrees (exact)',
  'dbscan-nodes': 'DBSCAN on tree nodes (approx.)',
  'dbscan-index': 'DBSCAN, conventional with index',
  'dbscan-baseline': 'DBSCAN, conventional without index',
};
const STRUCTURE_NAMES = {
  rstar: 'R*-tree', rquadratic: 'R-tree (quadratic)', rlinear: 'R-tree (linear)',
  str: 'R-tree (STR)', kdtree: 'k-d tree', quadtree: 'Quadtree',
};

// ---------------------------------------------------------------------------------------
// Theme: colours come from the CSS custom properties, so light and dark stay in one place.
// ---------------------------------------------------------------------------------------
let theme = null;
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  const series = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v('--series-' + i));
  const css = [...series, v('--noise'), v('--text-secondary')];
  theme = {
    css,
    rgba: css.map(hexToAbgr),
    surface: hexToAbgr(v('--surface-1')),
    ink: v('--text-primary'),
    ink2: v('--text-secondary'),
    accent: v('--accent'),
  };
}
function hexToAbgr(hex) {
  const n = parseInt(hex.slice(1), 16);
  return (0xff000000 | ((n & 0xff) << 16) | (n & 0xff00) | (n >> 16)) >>> 0;
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readTheme(); updateLegend(); draw(); });

// ---------------------------------------------------------------------------------------
// Server communication
// ---------------------------------------------------------------------------------------
function setBusy(delta) {
  S.busy += delta;
  $('busy').hidden = S.busy === 0;
}
async function api(path, body, raw) {
  setBusy(1);
  try {
    const opts = body === undefined ? {} : { method: 'POST', body: raw ? body : JSON.stringify(body) };
    const res = await fetch(path, opts);
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || res.statusText);
    return json;
  } finally {
    setBusy(-1);
  }
}
async function bin(path, Type) {
  const res = await fetch(path);
  if (!res.ok) throw new Error((await res.json()).error || res.statusText);
  return new Type(await res.arrayBuffer());
}
let toastTimer = 0;
function toast(msg) {
  $('toast').textContent = msg;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 6000);
}
/// Wraps an async UI action: errors are shown instead of being swallowed.
function act(fn) {
  return async (...args) => {
    try { await fn(...args); } catch (e) { toast(e.message); }
  };
}

// ---------------------------------------------------------------------------------------
// State handling
// ---------------------------------------------------------------------------------------
const dims = () => `dx=${$('dim-x').value || 0}&dy=${$('dim-y').value || 1}`;
const fmt = (x, digits = 3) => (x === null || x === undefined ? '–' : Number(x).toLocaleString('en-US', { maximumFractionDigits: digits }));

/// Called after the data set or the index changed on the server.
async function adopt(state, { keepView = false, dataChanged = true } = {}) {
  const firstTime = !S.state;
  const dChanged = firstTime || S.state.dataset.d !== state.dataset.d;
  S.state = state;
  S.selectedRun = null;
  S.runLabels = null;
  S.ari = {};
  S.query = null;
  const ds = state.dataset, ix = state.index;

  if (dChanged) {
    for (const [sel, def] of [[$('dim-x'), 0], [$('dim-y'), 1]]) {
      sel.innerHTML = '';
      for (let i = 0; i < ds.d; i++) sel.add(new Option('dimension ' + (i + 1), i));
      sel.value = def;
    }
    $('dims-row').hidden = ds.d <= 2;
  }
  $('data-info').textContent = `${ds.name}: ${fmt(ds.n)} points, ${ds.d} dimensions` + (ds.hasTruth ? ', with ground-truth labels' : '');
  $('index-info').textContent = `${STRUCTURE_NAMES[ix.structure]}: ${fmt(ix.nodes)} nodes, ${fmt(ix.leaves)} leaves, height ${ix.height}, built in ${fmt(ix.buildMs, 1)} ms`;
  $('idx-structure').value = ix.structure;
  $('idx-capacity').value = ix.capacity;
  $('color-by').querySelector('[value=truth]').disabled = !ds.hasTruth;
  if ($('color-by').value === 'truth' && !ds.hasTruth) $('color-by').value = 'partition';
  if ($('color-by').value === 'run') $('color-by').value = 'partition';

  if (dataChanged) {
    S.xy = await bin('/api/points?' + dims(), Float32Array);
    S.truth = ds.hasTruth ? await bin('/api/truth', Int32Array) : null;
    if (!keepView) {
      fitView();
      // a sensible eps for data of unknown scale: 2 % of the horizontal extent
      const dx = +$('dim-x').value;
      if (dChanged || firstTime || ds.name === 'upload') $('db-eps').value = +(0.02 * (ds.hi[dx] - ds.lo[dx])).toPrecision(2);
    }
  }
  const slider = $('depth');
  slider.max = ix.height;
  if (!keepView || S.depth > ix.height) S.depth = ix.height;  // the leaf level: the finest initial clustering
  slider.value = S.depth;
  renderRuns();
  await loadLevel();
}

/// Loads rectangles and point labels for the selected tree level.
async function loadLevel() {
  const depth = S.depth;
  const sizes = S.state.index.cutSizes;
  $('depth-label').textContent = `${depth} of ${S.state.index.height}: ${fmt(sizes[depth])} nodes` + (depth === S.state.index.height ? ' (leaves)' : depth === 0 ? ' (root)' : '');
  const [tree, part] = await Promise.all([
    api(`/api/tree?depth=${depth}&${dims()}`),
    bin(`/api/partition?depth=${depth}`, Int32Array),
  ]);
  if (depth !== S.depth) return;  // the slider moved on
  S.tree = tree.nodes;
  S.partition = part;
  applyColors();
}

function currentLabels() {
  switch ($('color-by').value) {
    case 'partition': return S.partition;
    case 'run': return S.runLabels;
    case 'truth': return S.truth;
    default: return null;
  }
}

function applyColors() {
  const n = S.xy ? S.xy.length / 2 : 0;
  const labels = currentLabels();
  const idx = new Uint8Array(n);
  const counts = new Map();
  if (!labels || labels.length !== n) {
    idx.fill(9);
  } else {
    for (let i = 0; i < n; i++) counts.set(labels[i], (counts.get(labels[i]) || 0) + 1);
    const many = counts.size - (counts.has(-1) ? 1 : 0) > 8;
    for (let i = 0; i < n; i++) {
      const l = labels[i];
      // Up to eight clusters take the palette in fixed order. More cannot be told apart by
      // hue; slots then repeat (scattered, so that neighbours usually differ).
      idx[i] = l < 0 ? 8 : many ? (Math.imul(l, 2654435761) >>> 16) % 8 : l % 8;
    }
  }
  S.colorIdx = idx;
  S.labelCounts = labels && labels.length === n ? counts : null;
  updateLegend();
  draw();
}

function updateLegend() {
  const el = $('legend');
  el.innerHTML = '';
  const counts = S.labelCounts;
  if (!counts) return;
  const mode = $('color-by').value;
  const title = document.createElement('b');
  const run = S.state.runs.find((r) => r.id === S.selectedRun);
  title.textContent = mode === 'partition' ? `Tree level ${S.depth}` : mode === 'truth' ? 'Ground truth' : run ? `Run ${run.id}: ${METHOD_NAMES[run.method]}` : '';
  el.append(title);
  const clusters = [...counts.keys()].filter((l) => l >= 0).sort((a, b) => a - b);
  const row = (slot, text) => {
    const d = document.createElement('div'), sw = document.createElement('i');
    sw.style.background = theme.css[slot];
    d.append(sw, text);
    el.append(d);
  };
  if (clusters.length <= 8) {
    for (const l of clusters) row(l % 8, `cluster ${l + 1}: ${fmt(counts.get(l))} points`);
  } else {
    const d = document.createElement('div');
    d.textContent = `${fmt(clusters.length)} clusters; colours repeat`;
    el.append(d);
  }
  if (counts.has(-1)) row(8, `noise: ${fmt(counts.get(-1))} points`);
}

// ---------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------
function fitView() {
  const ds = S.state.dataset, dx = +$('dim-x').value, dy = +$('dim-y').value;
  const w = Math.max(ds.hi[dx] - ds.lo[dx], 1e-12), h = Math.max(ds.hi[dy] - ds.lo[dy], 1e-12);
  const r = canvas.getBoundingClientRect();
  S.view.cx = (ds.lo[dx] + ds.hi[dx]) / 2;
  S.view.cy = (ds.lo[dy] + ds.hi[dy]) / 2;
  S.view.scale = 0.92 * Math.min(r.width / w, r.height / h);
}
const toScreen = (x, y) => [(x - S.view.cx) * S.view.scale + canvas.clientWidth / 2, canvas.clientHeight / 2 - (y - S.view.cy) * S.view.scale];
const toData = (sx, sy) => [(sx - canvas.clientWidth / 2) / S.view.scale + S.view.cx, (canvas.clientHeight / 2 - sy) / S.view.scale + S.view.cy];

let drawPending = false;
function draw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; render(); });
}

function render() {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(1, Math.round(canvas.clientWidth * dpr)), H = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  if (!theme || !S.xy) return;

  // Points are written straight into a pixel buffer: fast enough for millions of points.
  const img = ctx.createImageData(W, H);
  const px = new Uint32Array(img.data.buffer);
  px.fill(theme.surface);
  const n = S.xy.length / 2, xy = S.xy, ci = S.colorIdx, pal = theme.rgba;
  const size = Math.max(1, Math.round((n < 3000 ? 3 : n < 60000 ? 2 : 1) * dpr));
  const sc = S.view.scale * dpr, ox = W / 2 - S.view.cx * sc, oy = H / 2 + S.view.cy * sc, half = size >> 1;
  for (let i = 0; i < n; i++) {
    const x = Math.round(xy[2 * i] * sc + ox) - half, y = Math.round(oy - xy[2 * i + 1] * sc) - half;
    if (x < 0 || y < 0 || x + size > W || y + size > H) continue;
    const c = pal[ci[i]];
    for (let r = 0, o = y * W + x; r < size; r++, o += W)
      for (let q = 0; q < size; q++) px[o + q] = c;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.putImageData(img, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  drawRectangles();
  drawQuery();
}

function nodeRect(nd) {
  const [x0, y1] = toScreen(nd[2], nd[3]), [x1, y0] = toScreen(nd[4], nd[5]);
  return [x0, y0, Math.max(x1 - x0, 0.5), Math.max(y1 - y0, 0.5)];
}
const onCut = (nd) => nd[0] === S.depth || nd[1] === 1;

function drawRectangles() {
  const mode = $('show-mbr').value;
  if (mode === 'off') return;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const visible = (r) => r[0] < W && r[1] < H && r[0] + r[2] > 0 && r[1] + r[3] > 0;
  if (mode === 'all') {
    // the levels above the cut: recessive, heavier towards the root
    for (let depth = 0; depth < S.depth; depth++) {
      ctx.beginPath();
      for (const nd of S.tree) {
        if (nd[0] !== depth || nd[1] === 1) continue;
        const r = nodeRect(nd);
        if (visible(r)) ctx.rect(r[0], r[1], r[2], r[3]);
      }
      ctx.strokeStyle = theme.ink2;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = Math.max(1, 3 - depth);
      ctx.setLineDash([6, 4]);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  ctx.beginPath();
  for (const nd of S.tree) {
    if (!onCut(nd)) continue;
    const r = nodeRect(nd);
    if (visible(r)) ctx.rect(r[0], r[1], r[2], r[3]);
  }
  ctx.strokeStyle = theme.ink;
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawQuery() {
  const q = S.query;
  if (!q) return;
  // visited nodes: leaves filled, inner nodes outlined
  ctx.fillStyle = theme.accent;
  ctx.strokeStyle = theme.accent;
  ctx.lineWidth = 1.5;
  for (const v of q.visited) {
    const [x0, y1] = toScreen(v[0], v[1]), [x1, y0] = toScreen(v[2], v[3]);
    if (v[4]) { ctx.globalAlpha = 0.14; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
    ctx.globalAlpha = 0.8;
    ctx.strokeRect(x0, y0, Math.max(x1 - x0, 0.5), Math.max(y1 - y0, 0.5));
  }
  ctx.globalAlpha = 1;
  // result points
  const ids = q.ids, limit = Math.min(ids.length, 50000);
  ctx.fillStyle = theme.ink;
  for (let i = 0; i < limit; i++) {
    const [sx, sy] = toScreen(S.xy[2 * ids[i]], S.xy[2 * ids[i] + 1]);
    ctx.fillRect(sx - 2, sy - 2, 4, 4);
  }
  // the query itself
  ctx.strokeStyle = theme.ink;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  if (q.type === 'range') {
    const [x0, y0] = toScreen(q.x0, q.y0), [x1, y1] = toScreen(q.x1, q.y1);
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  } else {
    const [sx, sy] = toScreen(q.x, q.y);
    ctx.beginPath();
    ctx.arc(sx, sy, Math.max(q.radius * S.view.scale, 1), 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(sx - 6, sy); ctx.lineTo(sx + 6, sy); ctx.moveTo(sx, sy - 6); ctx.lineTo(sx, sy + 6);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

// ---------------------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------------------
function paramText(p) {
  return Object.entries(p).filter(([k]) => k !== 'method').map(([k, v]) => `${k}=${v}`).join(', ');
}
function workText(r) {
  const parts = [];
  if (r.iterations) parts.push(`${r.iterations} Lloyd iterations`);
  if (r.extra.micro_clusters !== undefined) parts.push(`${fmt(r.extra.micro_clusters)} nodes`);
  if (r.extra.range_queries !== undefined) parts.push(`${fmt(r.extra.range_queries)} eps-queries`);
  if (r.extra.dense_points !== undefined) parts.push(`${fmt(r.extra.dense_points)} points in ${fmt(r.extra.dense_nodes)} dense subtrees`);
  return parts.join(', ');
}

function renderRuns() {
  const body = $('runs').tBodies[0];
  body.innerHTML = '';
  if (!S.state.runs.length) {
    const td = body.insertRow().insertCell();
    td.colSpan = 13;
    td.className = 'empty';
    td.textContent = 'No runs yet. Use the buttons under “Cluster further”, or “Run all and compare”.';
    return;
  }
  for (const r of S.state.runs) {
    const tr = body.insertRow();
    tr.tabIndex = 0;
    if (r.id === S.selectedRun) tr.className = 'selected';
    const cell = (text, num) => { const td = tr.insertCell(); td.textContent = text; if (num) td.className = 'num'; };
    cell(r.id);
    cell(METHOD_NAMES[r.method] || r.method);
    cell(r.structure ? `${STRUCTURE_NAMES[r.structure]}, M=${r.capacity}` : 'none');
    cell(paramText(r.params));
    cell(fmt(r.clusters), true);
    cell(fmt(r.noise), true);
    cell(fmt(r.sse, 4), true);
    cell(fmt(r.ariTruth), true);
    cell(S.ari[r.id] === undefined ? '–' : fmt(S.ari[r.id]), true);
    cell(r.structure ? fmt(r.buildMs, 1) : '–', true);
    cell(fmt(r.ms, 1), true);
    cell(fmt(r.buildMs + r.ms, 1), true);
    cell(workText(r));
    const select = act(() => selectRun(r.id));
    tr.addEventListener('click', select);
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter') select(); });
  }
}

async function selectRun(id) {
  S.selectedRun = id;
  [S.runLabels, S.ari] = await Promise.all([bin('/api/labels?id=' + id, Int32Array), api('/api/ari?ref=' + id)]);
  $('color-by').value = 'run';
  renderRuns();
  applyColors();
}

async function runMethod(method, select = true) {
  const run = await api('/api/cluster', {
    method,
    depth: S.depth,
    k: +$('km-k').value,
    seed: +$('km-seed').value,
    eps: +$('db-eps').value,
    minPts: +$('db-minpts').value,
  });
  S.state.runs.push(run);
  if (select) await selectRun(run.id);
  else renderRuns();
  return run;
}

async function runAll() {
  const methods = ['index-partition', 'kmeans-baseline', 'kmeans-tree', 'kmeans-tree-seeded',
    'dbscan-index', 'dbscan-tree-dense', 'dbscan-nodes'];
  if (S.state.dataset.n <= 30000) methods.splice(4, 0, 'dbscan-baseline');
  let last = null;
  for (const m of methods) {
    try { last = await runMethod(m, false); } catch (e) { toast(`${METHOD_NAMES[m]}: ${e.message}`); }
  }
  if (last) await selectRun(last.id);
  $('runs').closest('.table-scroll').scrollTop = 1e9;
}

// ---------------------------------------------------------------------------------------
// Mouse interaction: pan/zoom, queries, insertion
// ---------------------------------------------------------------------------------------
function mousePos(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

async function runQuery(q) {
  const [dx, dy] = [+$('dim-x').value, +$('dim-y').value];
  const res = await api('/api/query', { ...q, dx, dy });
  S.query = { ...q, ids: res.ids, visited: res.stats.visited, radius: res.radius };
  const st = res.stats;
  const what = q.type === 'range' ? 'Range query' : q.type === 'knn' ? `${q.k} nearest neighbours` : `eps-ball (eps=${q.eps})`;
  $('query-info').textContent = `${what}: ${fmt(res.ids.length)} points. Visited ${fmt(st.nodesVisited)} of ${fmt(res.totalNodes)} nodes ` +
    `(${fmt(st.leavesVisited)} leaves), compared ${fmt(st.pointsChecked)} points, ${fmt(res.ms, 3)} ms.` +
    (S.state.dataset.d > 2 ? ' Other dimensions: unbounded for range queries, centre of the data space otherwise.' : '');
  draw();
}

async function insertAt(x, y) {
  const count = Math.max(1, Math.min(2000, +$('ins-count').value || 1));
  const spread = count === 1 ? 0 : 14 / S.view.scale;  // about 14 px standard deviation
  const points = [];
  for (let i = 0; i < count; i++) {
    // Box-Muller
    const r = Math.sqrt(-2 * Math.log(1 - Math.random())) * spread, t = 2 * Math.PI * Math.random();
    points.push([x + r * Math.cos(t), y + r * Math.sin(t)]);
  }
  const hadRuns = S.state.runs.length > 0;
  const state = await api('/api/insert', { points, dx: +$('dim-x').value, dy: +$('dim-y').value });
  await adopt(state, { keepView: true });
  $('query-info').textContent = `Inserted ${count} point${count > 1 ? 's' : ''} in ${fmt(state.insertMs, 3)} ms; the index now has ${fmt(state.index.nodes)} nodes.` +
    (hadRuns ? ' Earlier runs were discarded because they do not cover the new points.' : '');
}

canvas.addEventListener('mousedown', (e) => {
  const [sx, sy] = mousePos(e);
  S.drag = { sx, sy, cx: S.view.cx, cy: S.view.cy, moved: false };
});
window.addEventListener('mousemove', (e) => {
  const [sx, sy] = mousePos(e);
  const d = S.drag;
  if (d) {
    if (Math.abs(sx - d.sx) + Math.abs(sy - d.sy) > 3) d.moved = true;
    if (S.tool === 'range' && d.moved) {
      const [x0, y0] = toData(d.sx, d.sy), [x1, y1] = toData(sx, sy);
      S.query = { type: 'range', x0, y0, x1, y1, ids: [], visited: [] };
      draw();
    } else if (S.tool === 'pan' || d.moved) {
      if (S.tool !== 'range') {
        S.view.cx = d.cx - (sx - d.sx) / S.view.scale;
        S.view.cy = d.cy + (sy - d.sy) / S.view.scale;
        draw();
      }
    }
  }
  if (e.target === canvas) updateHover(sx, sy);
});
window.addEventListener('mouseup', act(async (e) => {
  const d = S.drag;
  if (!d) return;
  S.drag = null;
  const [sx, sy] = mousePos(e);
  const [x, y] = toData(sx, sy);
  if (S.tool === 'range' && d.moved) {
    const [x0, y0] = toData(d.sx, d.sy);
    await runQuery({ type: 'range', x0, y0, x1: x, y1: y });
  } else if (!d.moved) {
    if (S.tool === 'knn') await runQuery({ type: 'knn', x, y, k: Math.max(1, +$('q-k').value || 1) });
    else if (S.tool === 'radius') await runQuery({ type: 'radius', x, y, eps: +$('db-eps').value });
    else if (S.tool === 'insert') await insertAt(x, y);
    else if (S.query) { S.query = null; draw(); }
  }
}));
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const [sx, sy] = mousePos(e);
  const [x, y] = toData(sx, sy);
  S.view.scale *= Math.exp(-e.deltaY * 0.0015);
  // keep the point under the cursor fixed
  S.view.cx = x - (sx - canvas.clientWidth / 2) / S.view.scale;
  S.view.cy = y + (sy - canvas.clientHeight / 2) / S.view.scale;
  draw();
}, { passive: false });
canvas.addEventListener('dblclick', () => { if (S.tool === 'pan') { fitView(); draw(); } });
canvas.addEventListener('mouseleave', () => { $('hover').textContent = ''; });
canvas.addEventListener('keydown', (e) => {
  const step = 40 / S.view.scale;
  if (e.key === 'ArrowLeft') S.view.cx -= step;
  else if (e.key === 'ArrowRight') S.view.cx += step;
  else if (e.key === 'ArrowUp') S.view.cy += step;
  else if (e.key === 'ArrowDown') S.view.cy -= step;
  else if (e.key === '+' || e.key === '=') S.view.scale *= 1.25;
  else if (e.key === '-') S.view.scale /= 1.25;
  else if (e.key === '0') fitView();
  else return;
  e.preventDefault();
  draw();
});

/// Coordinates under the cursor and the node of the selected level that contains them.
function updateHover(sx, sy) {
  if (!S.state) return;
  const [x, y] = toData(sx, sy);
  let text = `x ${x.toPrecision(4)}  y ${y.toPrecision(4)}`;
  let hits = 0, node = null;
  for (const nd of S.tree) {
    if (!onCut(nd) || x < nd[2] || x > nd[4] || y < nd[3] || y > nd[5]) continue;
    hits++;
    if (!node || (nd[4] - nd[2]) * (nd[5] - nd[3]) < (node[4] - node[2]) * (node[5] - node[3])) node = nd;
  }
  if (node) {
    text += `  ·  ${node[1] ? 'leaf' : 'node'} at depth ${node[0]}, ${node[6]} ${node[1] ? 'points' : 'children'}`;
    if (hits > 1) text += `  ·  ${hits} overlapping nodes here`;
  }
  $('hover').textContent = text;
}

// ---------------------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------------------
$('gen-go').addEventListener('click', act(async () => {
  const state = await api('/api/dataset/generate', {
    kind: $('gen-kind').value, n: +$('gen-n').value, d: +$('gen-d').value, k: +$('gen-k').value,
    spread: +$('gen-spread').value, noise: +$('gen-noise').value, seed: +$('gen-seed').value,
  });
  $('km-k').value = $('gen-kind').value === 'moons' ? 2 : $('gen-k').value;
  await adopt(state);
}));
$('csv-file').addEventListener('change', act(async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const state = await api(`/api/dataset/upload?label=${$('csv-label').checked ? 1 : 0}&name=upload`, await file.text(), true);
  e.target.value = '';
  await adopt(state);
}));
$('idx-go').addEventListener('click', act(async () => {
  const state = await api('/api/index', { structure: $('idx-structure').value, capacity: +$('idx-capacity').value });
  await adopt(state, { dataChanged: false });
}));
for (const id of ['dim-x', 'dim-y']) {
  $(id).addEventListener('change', act(async () => {
    S.xy = await bin('/api/points?' + dims(), Float32Array);
    S.query = null;
    fitView();
    await loadLevel();
  }));
}
$('depth').addEventListener('input', act(async (e) => {
  S.depth = +e.target.value;
  if ($('color-by').value === 'run' || $('color-by').value === 'none') $('color-by').value = 'partition';
  await loadLevel();
}));
$('show-mbr').addEventListener('change', draw);
$('color-by').addEventListener('change', () => {
  if ($('color-by').value === 'run' && !S.runLabels) toast('Select a run in the table first.');
  applyColors();
});
$('record-partition').addEventListener('click', act(() => runMethod('index-partition')));
for (const b of document.querySelectorAll('button[data-method]')) b.addEventListener('click', act(() => runMethod(b.dataset.method)));
$('run-all').addEventListener('click', act(runAll));
$('clear-runs').addEventListener('click', act(async () => {
  const state = await api('/api/runs/clear', {});
  await adopt(state, { keepView: true, dataChanged: false });
}));
for (const b of document.querySelectorAll('button[data-tool]')) {
  b.addEventListener('click', () => {
    S.tool = b.dataset.tool;
    for (const o of document.querySelectorAll('button[data-tool]')) o.classList.toggle('active', o === b);
    canvas.style.cursor = S.tool === 'pan' ? 'grab' : 'crosshair';
  });
}
new ResizeObserver(draw).observe(canvas);

readTheme();
canvas.style.cursor = 'grab';
act(async () => { await adopt(await api('/api/state')); })();
