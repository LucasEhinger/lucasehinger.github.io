// The confusion matrices on /weather/details/ are drawn in the page, per
// (source, forecast lead), from confusion_by_lead.json.
//
// Three things this catches that a syntax check cannot. A runtime error in
// confusionPanel takes down the whole inline <script>, which also draws the
// feature chart and the results tables. A lead a source cannot reach must say so
// rather than render an empty or -- worse -- a misleadingly complete matrix. And
// the cell counts have to reconcile with the row totals the panel prints.
//
//   node scripts/test_confusion_by_lead.mjs
import fs from 'fs';

const html = fs.readFileSync('_pages/weather-details.html', 'utf8');
const data = JSON.parse(
  fs.readFileSync('files/weather/models/obs/confusion_by_lead.json', 'utf8'));

const start = html.indexOf('let CM_DATA = null;');
const end = html.indexOf('// Drawn from files/weather/models/obs/feature_importances.json');
if (start < 0 || end < 0) throw new Error('could not locate the confusion block');
const code = html.slice(start, end);

const mk = (tag) => ({
  tag, textContent: '', innerHTML: '', className: '', style: {}, children: [],
  appendChild(c) { this.children.push(c); return c; },
  insertAdjacentHTML(_, h) { this.innerHTML += h; },
});
global.document = { createElement: mk, getElementById: () => null };

const { confusionPanel, setData } = new Function('document', code + `
  ; return { confusionPanel, setData: (d) => { CM_DATA = d; } };
`)(global.document);

setData(data);
const LABELS = { all: 'Combined', hrrr: 'HRRR', nam: 'NAM', gfs: 'GFS',
                 rap: 'RAP', ecmwf: 'ECMWF', nbm: 'NBM' };
const LEADS = ['all', '1', '24', '48', '72', '96', '120', '144'];
const REACH = { hrrr: 48, rap: 51, nam: 60, gfs: 120, ecmwf: 144, nbm: 192, all: 144 };

const fail = [];
let panels = 0, matrices = 0, unreachable = 0;

for (const src of Object.keys(data.sources)) {
  for (const lead of LEADS) {
    let panel;
    try {
      panel = confusionPanel(src, lead, LABELS);
    } catch (e) {
      fail.push(`${src}/${lead}: threw ${e.message}`);
      continue;
    }
    panels += 1;
    const entry = data.sources[src][lead];
    const said = (panel.innerHTML || '') + panel.children.map(
      (c) => (c.innerHTML || '') + (c.textContent || '')).join('');

    if (!entry) {
      unreachable += 1;
      // Must explain itself, not render nothing.
      if (!/does not reach/.test(said)) {
        fail.push(`${src}/${lead}: absent, but the panel does not say the model cannot reach it`);
      }
      // And it should only be absent when the model genuinely stops short.
      if (lead !== 'all' && Number(lead) <= REACH[src]) {
        fail.push(`${src}/${lead}: missing although ${src} reaches ${REACH[src]} h`);
      }
      continue;
    }

    const grid = panel.children.find((c) => c.className === 'cm-wrap');
    if (!grid) { fail.push(`${src}/${lead}: no matrix grid rendered`); continue; }
    if (grid.children.length !== data.algorithms.length) {
      fail.push(`${src}/${lead}: ${grid.children.length} matrices, expected ${data.algorithms.length}`);
    }
    matrices += grid.children.length;

    // Arithmetic: the four cells must account for every row, for every algorithm.
    for (const algo of data.algorithms) {
      const c = entry.cells[algo];
      if (!c) { fail.push(`${src}/${lead}: no cells for ${algo}`); continue; }
      const total = c.tp + c.fp + c.fn + c.tn;
      if (total !== entry.n) {
        fail.push(`${src}/${lead}/${algo}: cells sum to ${total}, panel says ${entry.n} rows`);
      }
      if (c.tp + c.fn !== entry.positives) {
        fail.push(`${src}/${lead}/${algo}: tp+fn=${c.tp + c.fn}, positives=${entry.positives}`);
      }
    }

    // The vote must be a MAJORITY of the three hard calls, never an average: its
    // threshold field is null precisely to mark that it has no cut of its own.
    const vote = entry.cells['2 of 3 (the vote)'];
    if (vote && vote.threshold !== null) {
      fail.push(`${src}/${lead}: the vote reports a threshold (${vote.threshold}) — `
        + 'it is a majority of three calls and should not have one');
    }
  }
}

console.log(`${panels} panels, ${matrices} matrices, ${unreachable} lead(s) correctly `
  + 'reported as out of reach');
if (fail.length) {
  console.log('\nFAIL:');
  [...new Set(fail)].slice(0, 12).forEach((f) => console.log('  - ' + f));
  process.exit(1);
}
console.log('\nPASS: every panel renders, cells reconcile with the row totals, the vote '
  + 'carries no threshold, and unreachable leads explain themselves');
