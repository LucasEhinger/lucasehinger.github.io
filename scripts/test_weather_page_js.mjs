#!/usr/bin/env node
/**
 * Check the two bits of weather-page JavaScript that can fail silently.
 *
 * Both render into the page from data fetched at runtime, so a mistake does not
 * throw -- it produces a panel that is empty, or worse, one that looks right and
 * is wrong. Neither is visible in a diff.
 *
 *   1. wireFigurePicker (in _pages/weather-details.html) builds figure URLs from
 *      a source checkbox and, for feature importances, an algorithm radio. A
 *      toggle pointing at a file that does not exist is indistinguishable from a
 *      broken layout, so every URL it can build is checked against disk.
 *
 *   2. drawModelPerformance (in assets/js/weather-plots.js) renders precision,
 *      recall, ROC-AUC and a confusion matrix for the served model, reading the
 *      real metadata file. The matrix is the part worth testing: true negatives
 *      are DERIVED as n - tp - fp - fn, so an arithmetic slip would render a
 *      plausible-looking table that does not add up. Each lead is checked for
 *      internal consistency -- four cells summing to n, and precision and recall
 *      recomputed from the cells matching the stored values.
 *
 * Run from the repository root:
 *     node scripts/test_weather_page_js.mjs
 */
import fs from "fs";
import path from "path";

const ROOT = process.cwd();
const META = "files/weather/models/obs/model_metadata_all.json";
const ALGO = "Gradient Boosting";
let failures = 0;

function fail(msg) {
  console.log(`  FAIL ${msg}`);
  failures++;
}

/** Pull one function's source out of a file by brace matching. */
function extract(file, name) {
  const src = fs.readFileSync(file, "utf8");
  const start = src.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`${name} not found in ${file}`);
  let depth = 0;
  for (let i = src.indexOf("{", start); i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces in ${name}`);
}

/** Minimal DOM: enough for these two renderers, and nothing more. */
function makeDom() {
  const nodes = {};
  const made = [];
  function el(tag) {
    return {
      tag, textContent: "", className: "", type: "", name: "", value: "",
      checked: false, src: "", alt: "", loading: "", style: {}, attrs: {},
      children: [],
      appendChild(c) { this.children.push(c); if (c.tag === "img") made.push(c.src); },
      addEventListener(_e, fn) { this._fn = fn; },
      setAttribute(k, v) { this.attrs[k] = v; },
      getAttribute(k) { return this.attrs[k]; },
      querySelectorAll(sel) {
        const m = /input\[name="([^"]+)"\]:checked/.exec(sel);
        const all = [];
        (function walk(n) {
          (n.children || []).forEach((c) => { all.push(c); walk(c); });
        })(this);
        return m ? all.filter((c) => c.tag === "input" && c.name === m[1] && c.checked)
                 : all;
      },
      set innerHTML(_v) { this.children = []; },
    };
  }
  globalThis.document = {
    getElementById: (id) => (nodes[id] = nodes[id] || el("div")),
    createElement: el,
    createTextNode: (t) => ({ tag: "#text", textContent: t, appendChild() {} }),
  };
  return { nodes, made, el };
}

// --- 1. figure picker -------------------------------------------------------
function testFigurePicker() {
  console.log("wireFigurePicker -- every URL it builds must exist on disk");
  const src = extract("_pages/weather-details.html", "wireFigurePicker");
  const RESULT_MODELS = {
    all: "All parameters", hrrr: "HRRR", nam: "NAM", gfs: "GFS",
    rap: "RAP", ecmwf: "ECMWF", nbm: "NBM",
  };
  const { nodes, made } = makeDom();
  const fn = new Function("RESULT_MODELS", `${src}; return wireFigurePicker;`)(
    RESULT_MODELS
  );

  function urlsFor(args, flip) {
    made.length = 0;
    fn(...args);
    if (flip) {
      const algoEl = nodes[args[4]];
      const radios = algoEl.children.flatMap((w) =>
        (w.children || []).filter((c) => c.tag === "input")
      );
      const target = radios.find((r) => r.value === flip);
      if (!target) throw new Error(`no algorithm radio "${flip}"`);
      made.length = 0;
      target.checked = true;
      target._fn();
    }
    return [...made];
  }

  const cases = [
    ["source only (confusion matrices)",
     ["confusion-toggle", "confusion-figures", "confusion_matrices", "CM"], null],
    ["source + algorithm, default",
     ["features-toggle", "features-figures", "top_features", "TF", "features-algo"], null],
    ["after switching to XGBoost",
     ["features-toggle2", "features-figures2", "top_features", "TF", "features-algo2"],
     "xgboost"],
    ["after switching to Random Forest",
     ["features-toggle3", "features-figures3", "top_features", "TF", "features-algo3"],
     "random_forest"],
  ];
  for (const [label, args, flip] of cases) {
    const urls = urlsFor(args, flip);
    if (!urls.length) { fail(`${label}: produced no figures at all`); continue; }
    const missing = urls.filter((u) => !fs.existsSync(path.join(ROOT, u.replace(/^\//, ""))));
    if (missing.length) fail(`${label}: missing ${missing.join(", ")}`);
    else console.log(`  ok   ${label} -> ${urls.length} figure(s), all present`);
  }

  // Every (source, algorithm) pair the picker can reach must have a figure, not
  // just the defaults the cases above happen to hit.
  const algos = ["xgboost", "random_forest", "gradient_boosting"];
  const missing = [];
  for (const s of Object.keys(RESULT_MODELS)) {
    for (const a of algos) {
      const rel = `files/weather/examples/model_training_images/top_features_${s}_${a}.png`;
      if (!fs.existsSync(path.join(ROOT, rel))) missing.push(`${s}/${a}`);
    }
  }
  if (missing.length) fail(`missing feature figures for: ${missing.join(", ")}`);
  else console.log(`  ok   all ${Object.keys(RESULT_MODELS).length * algos.length} `
    + `(source, algorithm) feature figures present`);
}

// --- 2. model performance panel --------------------------------------------
function testModelPerformance() {
  console.log("\ndrawModelPerformance -- each lead's confusion matrix must add up");
  const src = extract("assets/js/weather-plots.js", "drawModelPerformance");
  const { nodes } = makeDom();
  const fn = new Function(`${src}; return drawModelPerformance;`)();
  const meta = JSON.parse(fs.readFileSync(path.join(ROOT, META), "utf8"))[ALGO];
  const byLead = meta.baserate_by_lead;
  fn(meta, byLead);

  const buttons = nodes["mp-lead"].children;
  const leads = Object.keys(byLead).map(Number).sort((a, b) => a - b);
  if (buttons.length !== leads.length) {
    fail(`${buttons.length} lead buttons for ${leads.length} leads`);
  }
  leads.forEach((lead, i) => {
    if (buttons[i] && buttons[i]._fn) buttons[i]._fn();
    const d = byLead[String(lead)];
    const rows = nodes["mp-cm"].children.map((tr) =>
      tr.children.map((c) => Number(String(c.textContent).replace(/,/g, "")))
    );
    const [tp, fnv] = [rows[1][1], rows[1][2]];
    const [fp, tn] = [rows[2][1], rows[2][2]];
    const sum = tp + fnv + fp + tn;
    const prec = tp / (tp + fp);
    const rec = tp / (tp + fnv);
    if (sum !== d.n) fail(`lead ${lead}h: cells sum to ${sum}, n is ${d.n}`);
    else if (Math.abs(prec - d.precision) > 0.002)
      fail(`lead ${lead}h: precision from cells ${prec.toFixed(4)} vs ${d.precision}`);
    else if (Math.abs(rec - d.recall) > 0.002)
      fail(`lead ${lead}h: recall from cells ${rec.toFixed(4)} vs ${d.recall}`);
    else
      console.log(`  ok   lead ${lead}h: tp=${tp} fp=${fp} fn=${fnv} tn=${tn}, `
        + `sums to n, precision and recall reproduce`);
  });
  if (!nodes["mp-note"].textContent.includes("held-out year")) {
    fail("the note under the matrix did not render");
  }
}

testFigurePicker();
testModelPerformance();
console.log(failures ? `\nFAILED (${failures})` : "\nPASS");
process.exit(failures ? 1 : 0);
