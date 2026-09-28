// The two capacity charts on /weather/details/ -- feature budget and tree depth
// -- are one function drawn twice. That sharing is the thing worth pinning: the
// charts must stay identical in construction while differing in data, baseline
// row, and axis label.
//
// The numbers are inlined in the page (they come from undercast_capacity.py's
// stdout, not from a JSON artifact), so the other job here is making sure the
// prose verdicts still follow from them -- "inside the noise band", "loses
// outside it", "no budget improves on the full model" are all claims that a
// retrain could silently invalidate.
//
//   node scripts/test_capacity_charts.mjs
import fs from 'fs';

const html = fs.readFileSync('_pages/weather-details.html', 'utf8');
let fail = 0;
const ok = (c, m) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fail++; };

// ---- DOM stub -----------------------------------------------------------
class El {
  constructor(t, ns) {
    this.tagName = t.toUpperCase(); this.ns = ns; this.attrs = {}; this.children = [];
    this.textContent = ''; this.style = {}; this.handlers = {}; this._cls = new Set();
    this.classList = { add: (c) => this._cls.add(c), remove: (c) => this._cls.delete(c) };
  }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get className() { return [...this._cls].join(' '); }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'class') this.className = v; }
  getAttribute(k) { return this.attrs[k] ?? null; }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener(t, fn) { (this.handlers[t] ||= []).push(fn); }
  fire(t) { (this.handlers[t] || []).forEach((f) => f()); }
  set innerHTML(v) { this._h = v; this.children = []; }
  get innerHTML() { return this._h || ''; }
  get all() { return this.children.flatMap((c) => [c, ...c.all]); }
}
const hosts = { 'trim-curve': new El('div'), 'depth-curve': new El('div') };
global.document = {
  getElementById: (id) => hosts[id] || null,
  createElement: (t) => new El(t),
  createElementNS: (ns, t) => new El(t, ns),
  createTextNode: (t) => { const e = new El('#text'); e.textContent = t; return e; },
};

const start = html.indexOf('        // One chart function, two charts.');
const end = html.indexOf('    </script>', start);
if (start < 0 || end < 0) throw new Error('could not locate the capacity-chart script');
const code = html.slice(start, end);
new Function(code)();

// recover the data the page drew from
const grab = (name) => JSON.parse(
  /\[[\s\S]*?\]\s*;/.exec(code.slice(code.indexOf(`var ${name} = `)))[0]
    .replace(/;$/, '').replace(/\n\s*/g, ''));
const WIDTH = grab('WIDTH_ROWS'), DEPTH = grab('DEPTH_ROWS');
const NOISE = 0.005;

// ---- both charts drew, from one function -------------------------------
ok(/function deltaChart\(/.test(code), 'a single deltaChart function is defined');
ok((code.match(/function deltaChart\(/g) || []).length === 1, 'defined exactly once, not copy-pasted');
for (const [id, rows] of [['trim-curve', WIDTH], ['depth-curve', DEPTH]]) {
  const host = hosts[id];
  const svg = host.children.find((c) => c.tagName === 'SVG');
  ok(Boolean(svg), `${id}: rendered`);
  const lines = svg.all.filter((e) => e._cls.has('lc-series'));
  const dots = svg.all.filter((e) => e._cls.has('lc-dot'));
  ok(lines.length === 4, `${id}: four series`);
  ok(dots.length === 4 * rows.length, `${id}: ${4 * rows.length} points`);
  ok(host.children.some((c) => c._cls.has('lc-legend')), `${id}: legend`);
}
const xlabels = ['trim-curve', 'depth-curve'].map((id) => hosts[id].children
  .find((c) => c.tagName === 'SVG').all
  .filter((e) => e._cls.has('lc-axis-label'))[0].textContent);
ok(xlabels[0] === 'features kept' && xlabels[1] === 'max depth',
  `axis labels differ per chart (${xlabels.join(' / ')})`);

// ---- the baseline row must be the one each chart claims ----------------
// The width sweep's baseline is its LAST row; the depth sweep's is in the
// middle. Getting that wrong would shift every delta silently.
const zeroDots = (id, rows, baseIdx) => {
  const svg = hosts[id].children.find((c) => c.tagName === 'SVG');
  const dots = svg.all.filter((e) => e._cls.has('lc-dot'));
  const ys = dots.map((dd) => Number(dd.getAttribute('cy')));
  // the four points at the baseline setting must all share one y (delta = 0)
  const atBase = [0, 1, 2, 3].map((k) => ys[k * rows.length + baseIdx]);
  return new Set(atBase.map((v) => v.toFixed(2))).size === 1;
};
ok(zeroDots('trim-curve', WIDTH, WIDTH.length - 1), 'width chart: all four series meet zero at 213');
ok(zeroDots('depth-curve', DEPTH, 2), 'depth chart: all four series meet zero at depth 3');

// ---- the prose verdicts must follow from the numbers -------------------
const deltas = (rows, baseIdx) => rows.map(
  (r) => [1, 2, 3, 4].map((i) => r[i] - rows[baseIdx][i]));

const wd = deltas(WIDTH, WIDTH.length - 1);
ok(!wd.some((row) => row.some((v) => v > NOISE)),
  '"No budget improves on the full model" holds: no gain beyond the noise floor');
const below70 = wd.slice(0, WIDTH.findIndex((r) => r[0] === 70));
ok(below70.every((row) => row.every((v) => v < -NOISE)),
  '"Below 70 features, every measure loses" holds');

const dd = deltas(DEPTH, 2);
const d2 = dd[DEPTH.findIndex((r) => r[0] === 2)];
ok(d2[0] <= NOISE && d2[2] <= NOISE,
  `depth 2's ROC gains are inside the noise band (${d2[0].toFixed(3)}, ${d2[2].toFixed(3)})`);
ok(d2[1] < -NOISE && d2[3] < -NOISE,
  `depth 2's PR losses are outside it (${d2[1].toFixed(3)}, ${d2[3].toFixed(3)})`);
// Prose checks are conditional, the same way test_redundancy_hist.mjs does it:
// requiring a sentence to still be present turns ordinary editing into a test
// failure. The invariant is the other direction -- a figure that IS quoted has
// to match the data.
const quoted = (re, what, ...expected) => {
  const m = re.exec(html);
  if (!m) { console.log(`   --   not stated on the page: ${what}`); return; }
  const got = m.slice(1).map(Number);
  ok(got.every((v, i) => Math.abs(v - Math.abs(expected[i])) < 5e-4),
    `${what}: page quotes ${got.join(' / ')}, data gives ${expected.map((e) => Math.abs(e).toFixed(3)).join(' / ')}`);
};
quoted(/loses PR-AUC by ([\d.]+) and ([\d.]+)/, 'depth 2 PR losses', d2[1], d2[3]);
const deeper = dd.filter((_, i) => DEPTH[i][0] > 3);
// The stronger claim -- beyond the reseed band -- is only enforced while the
// page makes it. The caption it does make is checked either way.
if (/Deeper is worse in every direction/.test(html)) {
  ok(deeper.every((row) => row.every((v) => v < -NOISE)),
    '"Deeper is worse in every direction" holds for depths 4 and 6');
} else {
  console.log('   --   not stated on the page: deeper-is-worse beyond the noise floor');
}
if (/degrades as one moves away from the current depth of 3/.test(html)) {
  ok(deeper.every((row) => row.every((v) => v < 0)),
    '"degrades as one moves away from the current depth of 3" holds for depths 4 and 6');
}
const d1 = dd[0];
quoted(/Stumps give up only ([\d.]+) ROC-AUC but ([\d.]+)\s+PR-AUC/, 'stumps claim', d1[0], d1[1]);

// depth 3 must actually hold the best PR-AUC on both holdouts, as claimed
const bestPR = (col) => DEPTH.reduce((a, b) => (b[col] > a[col] ? b : a))[0];
ok(bestPR(2) === 3 && bestPR(4) === 3,
  `"Depth 3 holds the best PR-AUC on both holdouts" holds (${bestPR(2)}, ${bestPR(4)})`);

console.log(fail ? `\n${fail} FAILED` : '\nOK');
process.exit(fail ? 1 : 0);
