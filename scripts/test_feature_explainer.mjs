// The "Top features" chart on /weather/details/ explains itself on hover.
//
// Two things can go wrong there and neither is visible from a syntax check.
// A runtime error in featurePanel takes down the whole inline <script>, which
// also draws the confusion matrices and the results tables. And a feature name
// the explainer does not recognise renders a bar whose tooltip says nothing
// useful -- which is the one job this chart has that a PNG could not do.
//
//   node scripts/test_feature_explainer.mjs
import fs from 'fs';

const html = fs.readFileSync('_pages/weather-details.html', 'utf8');
const data = JSON.parse(
  fs.readFileSync('files/weather/models/obs/feature_importances.json', 'utf8'));

// Pull the three pieces straight out of the page, so this tests what ships.
const start = html.indexOf('const FEAT_SOURCE = {');
const end = html.indexOf('// `renderPanel(key, algoSlug, ALGOS)`');
if (start < 0 || end < 0) throw new Error('could not locate the feature block in the page');
const code = html.slice(start, end);

// minimal DOM, same shape as test_undercast_panel.mjs
const mk = (tag) => ({
  tag, id: '', textContent: '', innerHTML: '', title: '', className: '', type: '',
  style: {}, children: [],
  appendChild(c) { this.children.push(c); return c; },
  append(...cs) { cs.forEach((c) => this.children.push(c)); },
  addEventListener(ev, fn) { (this._ev ||= {})[ev] = fn; },
  querySelector() { return null; },
});
global.document = { createElement: mk, getElementById: () => null };

const { featurePanel, explainFeature, setData } =
  new Function('document', code + `
    ; return { featurePanel, explainFeature, setData: (d) => { FEAT_DATA = d; } };
  `)(global.document);

setData(data);
const ALGOS = { gradient_boosting: 'Gradient Boosting', xgboost: 'XGBoost',
                random_forest: 'Random Forest' };
const LABELS = { all: 'Combined', hrrr: 'HRRR', nam: 'NAM', gfs: 'GFS',
                 rap: 'RAP', ecmwf: 'ECMWF', nbm: 'NBM' };

let panels = 0, rows = 0, fail = [];
for (const src of Object.keys(data)) {
  for (const algo of Object.keys(data[src])) {
    let panel;
    try {
      panel = featurePanel(src, algo, ALGOS, LABELS);
    } catch (e) {
      fail.push(`${src}/${algo}: threw ${e.message}`);
      continue;
    }
    panels += 1;
    const bars = panel.children.filter((c) => c.className === 'feat-row');
    if (bars.length !== data[src][algo].length) {
      fail.push(`${src}/${algo}: ${bars.length} bars for ${data[src][algo].length} features`);
    }
    bars.forEach((b) => {
      rows += 1;
      // Every bar must carry a plain-text tooltip AND update the panel's box.
      if (!b.title || b.title.length < 25) {
        fail.push(`${src}/${algo}: bar has no usable tooltip (${JSON.stringify(b.title)})`);
      }
      if (/<[^>]+>/.test(b.title)) {
        fail.push(`${src}/${algo}: tooltip contains raw HTML (${b.title.slice(0, 40)})`);
      }
      if (!b._ev || !b._ev.mouseenter || !b._ev.focus) {
        fail.push(`${src}/${algo}: bar is not wired for hover and keyboard focus`);
      }
    });
  }
}

// Nothing should fall through to the catch-all sentence: that is the explainer
// admitting it does not know, and it is the failure this file exists to catch.
const names = new Set();
for (const s of Object.keys(data))
  for (const a of Object.keys(data[s])) data[s][a].forEach((r) => names.add(r[0]));
const generic = [...names].filter(
  (n) => /a forecast field carried through from the source model/.test(explainFeature(n)));

console.log(`${panels} panels rendered, ${rows} bars, ${names.size} distinct feature names`);
if (generic.length) {
  console.log(`\n${generic.length} name(s) have no specific explanation:`);
  generic.forEach((n) => console.log('   ' + n));
  fail.push(`${generic.length} feature names fall back to the generic sentence`);
}
if (fail.length) {
  console.log('\nFAIL:');
  [...new Set(fail)].slice(0, 12).forEach((f) => console.log('  - ' + f));
  process.exit(1);
}
console.log('\nPASS: every bar renders, carries a plain-text tooltip, and is '
  + 'reachable by keyboard; every feature name has a specific explanation');
