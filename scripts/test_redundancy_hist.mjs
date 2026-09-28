// The cross-source correlation histogram on /weather/details/ is inline SVG
// drawn from redundancy_histogram.json, written by undercast_redundancy.py.
//
// It replaced a hand-written table of ten quantities. That table had the same
// weakness every hand-written table on this page has: the prose around it
// quotes numbers, and nothing checks that they still match the artifact after a
// retrain. So the main job here is drift -- every figure the surrounding
// paragraphs claim is asserted against the JSON.
//
//   node scripts/test_redundancy_hist.mjs
import fs from 'fs';

const html = fs.readFileSync('_pages/weather-details.html', 'utf8');
const d = JSON.parse(
  fs.readFileSync('files/weather/models/obs/redundancy_histogram.json', 'utf8'));

let fail = 0;
const ok = (c, m) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fail++; };
const near = (a, b, t = 5e-4) => Math.abs(a - b) <= t;

// ---- 1. the JSON is internally consistent ------------------------------
const sum = (a) => a.reduce((x, y) => x + y, 0);
ok(d.edges.length === d.same.counts.length + 1, 'one more edge than bins');
ok(sum(d.same.counts) === d.same.n, `same-quantity counts sum to n (${d.same.n})`);
ok(sum(d.baseline.counts) === d.baseline.n, `baseline counts sum to n (${d.baseline.n})`);
ok(near(sum(d.same.share), 1) && near(sum(d.baseline.share), 1), 'both shares sum to 1');

// The prose still quotes "47% exceed 0.9", so a bin edge has to land on 0.9 for
// that figure to be recoverable from what the chart draws.
const hiIdx = d.edges.findIndex((e) => near(e, 0.9, 1e-9));
ok(hiIdx > 0, 'a bin edge lands exactly on 0.9');
const above09 = sum(d.same.counts.slice(hiIdx)) / d.same.n;
ok(near(above09, d.same.share_above_0_9, 1e-3),
  `the >0.9 share matches the bins (${(100 * above09).toFixed(1)}%)`);

// ---- 2. every number the page claims, against the artifact -------------
// Conditional on purpose. Requiring the prose to CONTAIN each sentence makes the
// test fail the moment the author edits the copy, which is not a defect -- the
// invariant worth holding is the other direction: if a figure is stated, it has
// to match the artifact. A claim that is simply gone is reported, not failed.
const claims = [
  ['300 same-quantity pairs', /All (\d[\d,]*)\s+same-quantity, different-source pairs/, () => d.same.n],
  ['baseline pair count', /([\d,]+)\s+randomly\s+drawn pairs of/, () => d.baseline.n],
  ['same-quantity median', /median is <strong>([\d.]+)<\/strong>/, () => d.same.median],
  ['baseline median', /<strong>([\d.]+)<\/strong> for a randomly drawn pair/, () => d.baseline.median],
  ['share above 0.9', /(?:Forty-\s*seven|47) percent of the pairs exceed 0\.9/, () => 47,
    () => Math.round(100 * d.same.share_above_0_9)],
  ['share above 0.95', /(?:thirty-seven|37) percent exceed 0\.95/, () => 37,
    () => Math.round(100 * d.same.share_above_0_95)],
  ['count in the top bin', /(\d+) of the 300\s+pairs sit above 0\.95/, () => d.same.counts[d.same.counts.length - 1]],
];
let stated = 0;
for (const [what, re, expected, fixed] of claims) {
  const m = re.exec(html);
  if (!m) { console.log(`   --   not stated on the page: ${what}`); continue; }
  stated++;
  const want = expected();
  // A regex with no capture group is a fixed phrase ("Forty-seven percent"); the
  // artifact value is then checked against the number that phrase spells out.
  const got = m[1] !== undefined ? Number(String(m[1]).replace(/,/g, '')) : fixed();
  ok(near(got, want, 5e-4), `${what}: page says ${got}, artifact says ${typeof want === 'number' && !Number.isInteger(want) ? want.toFixed(3) : want}`);
}
ok(stated > 0, `${stated} of ${claims.length} claims are stated and were checked`);

// ---- 3. the chart draws -------------------------------------------------
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
  set innerHTML(v) { this._h = v; this.children = []; }
  get innerHTML() { return this._h || ''; }
  get all() { return this.children.flatMap((c) => [c, ...c.all]); }
}
const host = new El('div');
global.document = {
  getElementById: (id) => (id === 'redundancy-hist' ? host : null),
  createElement: (t) => new El(t),
  createElementNS: (ns, t) => new El(t, ns),
  createTextNode: (t) => { const e = new El('#text'); e.textContent = t; return e; },
};
let fetched = null;
global.fetch = (u) => { fetched = u; return Promise.resolve({ ok: true, json: () => Promise.resolve(d) }); };

const start = html.indexOf('        // Self-contained: this section sits below');
const end = html.indexOf('    </script>', start);
if (start < 0 || end < 0) throw new Error('could not locate the histogram script');
new Function(html.slice(start, end))();
await new Promise((r) => setTimeout(r, 0));

ok(fetched === '/files/weather/models/obs/redundancy_histogram.json',
  `fetches the exported artifact (${fetched})`);
const svg = host.children.find((c) => c.tagName === 'SVG');
ok(Boolean(svg), 'an <svg> was rendered');
const bars = svg.all.filter((e) => e._cls.has('rh-bar'));
ok(bars.length === d.same.counts.length, `${d.same.counts.length} bars, one per bin`);

// One fill for every bar: the <0.9 / >=0.9 split was removed deliberately, so a
// second colour reappearing here means it crept back.
const fills = new Set(bars.map((b) => b.getAttribute('fill')));
ok(fills.size === 1, `all bars share one fill (${[...fills].join(', ')})`);

// bar heights must be proportional to share, or the shape is a lie
const heights = bars.map((b) => Number(b.getAttribute('height')));
// Calibrate the px-per-share scale off the tallest bar, then require every
// other bar to sit on that same line -- a bar drawn from the wrong bin would
// still have a plausible height, but not a proportional one.
const scale = heights[heights.length - 1] / d.same.share[d.same.share.length - 1];
const proportional = heights.every((h, i) => Math.abs(h - scale * d.same.share[i]) < 0.6);
ok(proportional, 'every bar height is proportional to its share');

ok(svg.all.some((e) => e._cls.has('rh-base')), 'the baseline outline is drawn');
ok(host.children.some((c) => c._cls.has('lc-legend')), 'a legend is rendered');
const readout = host.children.find((c) => c._cls.has('lc-readout'));
ok(readout && /300 same-quantity pairs/.test(readout.textContent),
  'the readout states the sample size');

// ---- 4. theme safety ----------------------------------------------------
const css = html.slice(html.indexOf('.rh-bar {'), html.indexOf('.toc-sidebar {'));
const inks = [...css.matchAll(/(stroke|color|background)\s*:\s*([^;]+);/g)]
  .map((m) => m[2].trim()).filter((v) => /^#|^rgb|^black$|^white$/i.test(v));
ok(inks.length === 0, `no hard-coded ink in the chart's structural CSS${inks.length ? `: ${inks}` : ''}`);

console.log(fail ? `\n${fail} FAILED` : '\nOK');
process.exit(fail ? 1 : 0);
