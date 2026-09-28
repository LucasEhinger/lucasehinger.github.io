// The skill-vs-lead chart on /weather/details/ is inline SVG built at runtime
// from model_metadata_<source>.json. Four things this catches that eyeballing
// one metric in one theme does not:
//
//   * a runtime error here takes down the whole shared <script>, which also
//     draws the confusion matrices, the feature chart and the results tables;
//   * a source must STOP at its reach, not plot zero -- a 0 would draw a cliff
//     to the axis and read as "the model got everything wrong at 96 h";
//   * every plotted value has to match the artifact it claims to come from;
//   * every colour must come from a CSS variable or the palette, never a
//     hard-coded ink that vanishes in dark mode.
//
//   node scripts/test_lead_curves.mjs
import fs from 'fs';

const html = fs.readFileSync('_pages/weather-details.html', 'utf8');
const SOURCES = ['all', 'hrrr', 'rap', 'nam', 'gfs', 'ecmwf', 'nbm'];
const LEADS = [1, 24, 48, 72, 96, 120, 144];
const ALGO = 'Gradient Boosting';

const meta = Object.fromEntries(SOURCES.map((s) => [
  s, JSON.parse(fs.readFileSync(`files/weather/models/obs/model_metadata_${s}.json`, 'utf8')),
]));

let fail = 0;
const ok = (c, m) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fail++; };

// ---- DOM stub ----------------------------------------------------------
class El {
  constructor(tag, ns) {
    this.tagName = tag.toUpperCase(); this.ns = ns || null;
    this.attrs = {}; this.children = []; this.parent = null;
    this.textContent = ''; this.style = {}; this.handlers = {};
    this._cls = new Set();
    this.classList = {
      add: (...c) => c.forEach((x) => this._cls.add(x)),
      remove: (...c) => c.forEach((x) => this._cls.delete(x)),
      toggle: (c, on) => (on ? this._cls.add(c) : this._cls.delete(c)),
      contains: (c) => this._cls.has(c),
    };
  }
  get className() { return [...this._cls].join(' '); }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  setAttribute(k, v) {
    this.attrs[k] = String(v);
    if (k === 'class') this.className = v;
  }
  getAttribute(k) { return k === 'class' ? this.className : (this.attrs[k] ?? null); }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  addEventListener(t, fn) { (this.handlers[t] ||= []).push(fn); }
  fire(t) { (this.handlers[t] || []).forEach((fn) => fn()); }
  set innerHTML(v) { this._html = v; this.children = []; }
  get innerHTML() { return this._html || ''; }
  get all() { return this.children.flatMap((c) => [c, ...c.all]); }
  querySelectorAll(sel) {
    return this.all.filter((e) => sel.split(',').map((x) => x.trim()).some((s) => (
      s.startsWith('[') ? e.getAttribute(s.slice(1, -1)) !== null
        : s.includes('.') ? s.split('.').filter(Boolean).every((c, i) => (i === 0 && !s.startsWith('.') ? e.tagName === c.toUpperCase() : e._cls.has(c)))
          : e.tagName === s.toUpperCase())));
  }
  querySelector(s) { return this.querySelectorAll(s)[0] || null; }
}

const picker = new El('div');
const host = new El('div');
const byId = { 'leadcurve-metric': picker, leadcurve: host };
global.document = {
  getElementById: (id) => byId[id] || null,
  createElement: (t) => new El(t),
  createElementNS: (ns, t) => new El(t, ns),
  createTextNode: (t) => { const e = new El('#text'); e.textContent = t; return e; },
};
const metadataCache = {};
const metadataUrl = (k) => k;
global.fetch = (k) => Promise.resolve({ ok: true, json: () => Promise.resolve(meta[k]) });

// ---- run the page's real code -----------------------------------------
// The call site matters as much as the function. This <script> block sits
// ABOVE the #leadcurve markup, so calling wireLeadCurves at parse time is a
// silent no-op: getElementById returns null, it returns early, and the page
// shows an empty picker and no chart. Calling the function directly, the way
// this test does, would never notice -- so pin the ordering explicitly.
const markupAt = html.indexOf('<div class="lc-wrap" id="leadcurve">');
const callAt = html.indexOf('wireLeadCurves("leadcurve-metric", "leadcurve")', html.indexOf('function wireLeadCurves'));
const deferred = /readyState === "loading"[\s\S]{0,240}wireLeadCurves\("leadcurve-metric"/.test(html);
ok(markupAt > 0 && callAt > 0, 'both the chart markup and its call site exist');
ok(markupAt < callAt || deferred,
  'the chart is wired after its markup exists (markup first, or deferred to DOMContentLoaded)');

const start = html.indexOf('// ---- skill against forecast lead');
const end = html.indexOf('            // Deferred, unlike the other pickers');
if (start < 0 || end < 0) throw new Error('could not locate the chart block');
const run = new Function('metadataCache', 'metadataUrl',
  `${html.slice(start, end)}\nreturn { wireLeadCurves, LC_METRICS, LC_COLORS };`);
const { wireLeadCurves, LC_METRICS, LC_COLORS } = run(metadataCache, metadataUrl);

wireLeadCurves('leadcurve-metric', 'leadcurve');
await new Promise((r) => setTimeout(r, 0));

const metrics = Object.keys(LC_METRICS);
ok(metrics.length === 4, `four metrics offered (${metrics.join(', ')})`);
ok(picker.querySelectorAll('input').length === 4, 'one radio per metric');
ok(picker.querySelectorAll('input').filter((i) => i.attrs.checked || i.checked).length <= 1,
  'at most one metric preselected');

// ---- every metric draws, and matches the artifacts ---------------------
let checked = 0;
for (const metric of metrics) {
  const radio = picker.querySelectorAll('input').find((i) => i.value === metric);
  radio.value = metric; radio.fire('change');

  const svg = host.children.find((c) => c.tagName === 'SVG');
  if (!svg) { ok(false, `${metric}: chart rendered`); continue; }
  const paths = svg.querySelectorAll('path');
  const dots = svg.querySelectorAll('circle');

  // reach: a source's line must have exactly as many points as it has leads
  let reachOk = true, valueOk = true;
  for (const src of SOURCES) {
    const by = meta[src][ALGO].baserate_by_lead;
    const expect = LEADS.filter((L) => typeof by[String(L)]?.[metric] === 'number');
    const got = dots.filter((d) => d.getAttribute('data-src') === src);
    if (got.length !== expect.length) {
      reachOk = false;
      console.log(`      ${metric}/${src}: ${got.length} points, expected ${expect.length}`);
    }
    // the <title> on each dot quotes the value; check it against the artifact
    for (const d of got) {
      const t = d.children.find((c) => c.tagName === 'TITLE');
      const m = /at (\d+) h — .* ([\d.]+)$/.exec(t.textContent);
      if (!m) { valueOk = false; continue; }
      // Compare the RENDERED string, not a tolerance: the label is what the
      // reader sees, and a tolerance of half the last digit is ambiguous at
      // exact half-steps (0.1875 -> "0.188").
      const want = by[m[1]][metric].toFixed(3);
      if (m[2] !== want) {
        valueOk = false;
        console.log(`      ${metric}/${src}@${m[1]}h: plotted ${m[2]}, artifact ${want}`);
      }
      checked++;
    }
  }
  ok(paths.length === SOURCES.length, `${metric}: one line per source`);
  ok(reachOk, `${metric}: each line stops at its source's reach`);
  ok(valueOk, `${metric}: every plotted point matches the artifact`);

  // y-axis must not clip the data
  const ticks = svg.querySelectorAll('text').filter((t) => t._cls.has('lc-tick'))
    .map((t) => Number(t.textContent)).filter((n) => !Number.isNaN(n) && n <= 1);
  const vals = SOURCES.flatMap((s) => LEADS
    .map((L) => meta[s][ALGO].baserate_by_lead[String(L)]?.[metric])
    .filter((v) => typeof v === 'number'));
  ok(Math.max(...ticks) >= Math.max(...vals) - 1e-9,
    `${metric}: axis top ${Math.max(...ticks)} covers max ${Math.max(...vals).toFixed(3)}`);
}
ok(checked > 100, `${checked} plotted values cross-checked against the artifacts`);

// ---- no point is ever plotted as a fake zero ---------------------------
const zeroCliff = SOURCES.some((s) => {
  const by = meta[s][ALGO].baserate_by_lead;
  return LEADS.some((L) => by[String(L)] && by[String(L)].roc_auc === 0);
});
ok(!zeroCliff, 'no source reports a real zero that the "absent" rule would swallow');

// ---- theme safety ------------------------------------------------------
const cssStart = html.indexOf('.lc-wrap {');
const cssEnd = html.indexOf('.figure-grid {');
const css = html.slice(cssStart, cssEnd);
const inks = [...css.matchAll(/(fill|stroke|color|background)\s*:\s*([^;]+);/g)]
  .map((m) => [m[1], m[2].trim()])
  .filter(([, v]) => /^#|^rgb|^black$|^white$/i.test(v));
ok(inks.length === 0, `no hard-coded ink in the chart CSS${inks.length ? `: ${JSON.stringify(inks)}` : ''}`);
const palette = new Set(Object.values(LC_COLORS));
ok(palette.size === SOURCES.length, 'every source has its own colour');

console.log(fail ? `\n${fail} FAILED` : '\nOK');
process.exit(fail ? 1 : 0);
