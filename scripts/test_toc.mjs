// The contents list on /weather/details/ is built at runtime, and then a second
// copy of it takes over the left sidebar once the inline list scrolls away.
//
// Two bugs this pins. (1) The builder used to query the whole document, which
// swept in the theme's author sidebar -- <h3 class="author__name">Lucas
// Ehinger</h3> -- and since that heading precedes every <h2>, it became the
// first numbered entry. (2) The sidebar copy is a cloneNode of a list whose root
// carries id="toc-list"; shipping that clone unstripped puts a duplicate id in
// the document, which quietly breaks getElementById for everything downstream.
//
// It also checks the swap only happens where the theme makes .sidebar fixed
// (>=1024px). Below that the sidebar is a normal block ABOVE the article, so
// swapping would hide the author card and put a second contents list exactly
// where the reader just scrolled past the first.
//
//   node scripts/test_toc.mjs
import fs from 'fs';

const PAGE = process.env.TOC_PAGE || '_pages/weather-details.html';
const BUILT = '_site/weather/details/index.html';
const html = fs.readFileSync(PAGE, 'utf8');

let fail = 0;
const ok = (c, m) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fail++; };

// ---- 1. the fixture must describe the real page ------------------------
if (fs.existsSync(BUILT)) {
  const built = fs.readFileSync(BUILT, 'utf8');
  for (const [re, what] of [
    [/<h3 class="author__name">/, 'author sidebar heading (the bug source)'],
    [/<div class="author__content">/, 'author__content wrapper'],
    [/<div class="sidebar sticky">/, 'sidebar wrapper the swap targets'],
    [/<div class="archive">/, 'article wrapper the TOC scopes to'],
  ]) if (!re.test(built)) throw new Error(`built site no longer has ${what}; update this fixture`);
  if (built.indexOf('author__name') > built.indexOf('<div class="archive">'))
    throw new Error('sidebar no longer precedes the article');
  console.log('fixture matches the built site');
} else {
  console.log('note: _site not built, skipping the fixture cross-check');
}
// the CSS guard and the script guard must name the same breakpoint
const cssBp = /@media screen and \(max-width: (\d+)px\)[^{]*\{\s*\.toc-sidebar/.exec(html);
const jsBp = /matchMedia\('\(min-width: (\d+)px\)'\)/.exec(html);
ok(cssBp && jsBp && Number(cssBp[1]) + 1 === Number(jsBp[1]),
  `CSS (<=${cssBp?.[1]}px) and script (>=${jsBp?.[1]}px) guards agree`);
const themeBp = fs.existsSync('_sass/theme/_air_light.scss')
  && /\$sidebar-screen-min-width\s*:\s*(\d+)px/.exec(fs.readFileSync('_sass/theme/_air_light.scss', 'utf8'));
ok(!themeBp || themeBp[1] === jsBp[1],
  `breakpoint matches the theme's $sidebar-screen-min-width (${themeBp ? themeBp[1] : '?'}px)`);

// ---- 2. minimal DOM ----------------------------------------------------
class El {
  constructor(tag, cls = '', text = '') {
    this.tagName = tag.toUpperCase(); this.attrs = cls ? { class: cls } : {};
    this.textContent = text; this.children = []; this.parent = null;
    this.hidden = false; this.handlers = {};
    this._cls = new Set(cls.split(/\s+/).filter(Boolean));
    this.classList = {
      add: (c) => this._cls.add(c), remove: (c) => this._cls.delete(c),
      toggle: (c, on) => (on ? this._cls.add(c) : this._cls.delete(c)),
      contains: (c) => this._cls.has(c),
    };
  }
  get className() { return [...this._cls].join(' '); }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); this.attrs.class = v; }
  get id() { return this.attrs.id || ''; }
  set id(v) { this.attrs.id = v; }
  // Real anchors reflect href between property and attribute, and the builder
  // assigns the property while the sidebar clone reads the attribute back.
  get href() { return this.attrs.href || ''; }
  set href(v) { this.attrs.href = v; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'class') this._cls = new Set(v.split(/\s+/)); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  cloneNode() {
    const c = new El(this.tagName, this.className, this.textContent);
    c.attrs = { ...this.attrs };
    this.children.forEach((k) => c.appendChild(k.cloneNode(true)));
    return c;
  }
  get all() { return this.children.flatMap((c) => [c, ...c.all]); }
  _match(sel) {
    sel = sel.trim();
    if (sel.startsWith('[') && sel.endsWith(']')) return this.getAttribute(sel.slice(1, -1)) !== null;
    const [tag, ...cls] = sel.split('.');
    if (tag && this.tagName !== tag.toUpperCase()) return false;
    return cls.every((c) => this._cls.has(c));
  }
  querySelectorAll(sel) { return this.all.filter((e) => sel.split(',').some((s) => e._match(s))); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  contains(e) { for (let p = e; p; p = p.parent) if (p === this) return true; return false; }
  closest(sel) {
    for (let p = this; p; p = p.parent) if (sel.split(',').some((s) => p._match(s))) return p;
    return null;
  }
}

const root = new El('body');
const sidebar = root.appendChild(new El('div', 'sidebar sticky'));
const authorCard = sidebar.appendChild(new El('div', ''));
authorCard.appendChild(new El('div', 'author__content'))
  .appendChild(new El('h3', 'author__name', 'Lucas Ehinger'));

const archive = root.appendChild(new El('div', 'archive'));
const nav = archive.appendChild(new El('nav')); nav.id = 'toc';
nav.appendChild(new El('h2', '', 'Contents'));
const list = nav.appendChild(new El('ol')); list.id = 'toc-list';

const HEADINGS = [
  ['h2', 'About Undercasts'], ['h3', 'Cloud layers'],
  ['h2', 'Building a Forecast'], ['h3', 'Top features'],
  ['h3', 'Skill against forecast lead'],
  ['h3', 'Which ML model is best? (click to expand)'],
];
// One heading lives inside a <details>, the way "Input parameters" does. It has
// to be appended IN DOCUMENT ORDER or the builder's querySelectorAll sees a
// different sequence than the page would.
const headEls = HEADINGS.map(([t, x], i) => {
  const parent = i === 4 ? archive.appendChild(new El('details')) : archive;
  return parent.appendChild(new El(t, '', x));
});

// Live lookup, not a snapshot: the builder assigns ids to headings as it goes
// and the sidebar clone then resolves them back by href.
const findById = (id) => root.all.find((e) => e.id === id) || (root.id === id ? root : null) || null;
global.document = {
  getElementById: findById,
  createElement: (t) => new El(t),
  querySelector: (s) => root.querySelector(s),
  querySelectorAll: (s) => root.querySelectorAll(s),
};

let mqWide = true;
const mqListeners = [];
global.window = {
  matchMedia: () => ({
    get matches() { return mqWide; },
    addEventListener: (_, fn) => mqListeners.push(fn),
  }),
};
const observers = [];
global.IntersectionObserver = class {
  constructor(cb, opts) { this.cb = cb; this.opts = opts; this.targets = new Set(); observers.push(this); }
  observe(t) { this.targets.add(t); }
};
global.window.IntersectionObserver = global.IntersectionObserver;

// ---- 3. run the page's real builder ------------------------------------
const start = html.indexOf("var list = document.getElementById('toc-list');");
const end = html.lastIndexOf('    })();');
if (start < 0 || end < 0) throw new Error('could not locate the TOC builder');
new Function(html.slice(start, end))();

// ---- 4. the inline list -------------------------------------------------
const entries = [];
(function walk(ol, depth) {
  for (const li of ol.children) {
    const a = li.children.find((c) => c.tagName === 'A');
    if (a) entries.push({ text: a.textContent, href: a.getAttribute('href'), depth, el: a });
    const sub = li.children.find((c) => c.tagName === 'OL');
    if (sub) walk(sub, depth + 1);
  }
})(list, 0);

ok(!entries.some((e) => /Lucas Ehinger/.test(e.text)), 'author name is not in the contents');
ok(entries[0]?.text === 'About Undercasts', `first entry is "About Undercasts" (got "${entries[0]?.text}")`);
ok(entries[0]?.depth === 0, 'first entry is top-level');
ok(!entries.some((e) => /Contents/.test(e.text)), "the TOC's own heading is excluded");
ok(entries.length === HEADINGS.length, `all ${HEADINGS.length} article headings listed (got ${entries.length})`);
ok(entries.some((e) => e.text === 'Which ML model is best?'), '"(click to expand)" hint stripped');

// ---- 4b. section rules ---------------------------------------------------
// Same heading list as the contents, so they cannot disagree about what a
// section is. The skips are the whole point: no rule on the first heading, none
// on an h3 that opens its h2's section, none inside a <details>.
const ruled = headEls.map((h) => (
  h._cls.has('section-rule') ? 'full' : h._cls.has('section-rule-sub') ? 'light' : null));
// Fixture order: h2, h3(opens it), h2, h3(opens it), h3(in <details>), h3.
const EXPECT = [null, null, 'full', null, null, 'light'];
ok(JSON.stringify(ruled) === JSON.stringify(EXPECT),
  `section rules land where expected (got ${JSON.stringify(ruled)})`);
ok(ruled[0] === null, 'no rule above the first heading');
ok(ruled[1] === null && ruled[3] === null, 'no rule on an h3 that opens its h2 section');
ok(ruled[2] === 'full', 'h2 sections get the full rule');
ok(ruled[5] === 'light', 'a later h3 gets the light rule');
ok(!headEls.some((h, i) => ruled[i] && h.closest('details')),
  'no rule on a heading inside <details>');
ok(new Set(['section-rule', 'section-rule-sub']).size === 2
  && /\.section-rule\s*\{/.test(html) && /\.section-rule-sub::before/.test(html),
  'both rule styles are defined in the page CSS');

// ---- 5. the sidebar copy ------------------------------------------------
const panel = sidebar.querySelector('div.toc-sidebar');
ok(Boolean(panel), 'a contents panel was added to the sidebar');
const copyLinks = panel ? panel.querySelectorAll('a') : [];
ok(copyLinks.length === entries.length, `sidebar copy has all ${entries.length} entries`);
ok(copyLinks.every((a, i) => a.getAttribute('href') === entries[i].href),
  'sidebar copy links to the same anchors');

const ids = root.all.map((e) => e.id).filter(Boolean);
ok(new Set(ids).size === ids.length, `no duplicate ids after cloning (${ids.length} ids)`);
ok(panel && panel.querySelectorAll('[id]').length === 0, 'the clone carries no ids at all');

// ---- 5b. sidebar type sizes must not compound ---------------------------
// The theme's `.sidebar p, .sidebar li { font-size: .75em }` is relative and
// applies at every level, so an em-based size here multiplies down the nesting
// and sub-entries end up around 9.6px. Both levels must be declared in an
// absolute unit, on the <li> itself -- a size on the container loses outright,
// because a rule on the element beats inheritance.
const tocCss = html;   // both selectors are unique; no slice to get out of order
const topSize = /\.toc-sidebar>ol>li\s*\{[^}]*font-size:\s*([\d.]+)(r?em)/.exec(tocCss);
const subSize = /\.toc-sidebar ol ol li\s*\{[^}]*font-size:\s*([\d.]+)(r?em)/.exec(tocCss);
ok(topSize && topSize[2] === 'rem',
  `top-level entries sized in rem (got ${topSize ? topSize[1] + topSize[2] : 'nothing'})`);
ok(subSize && subSize[2] === 'rem',
  `sub-entries sized in rem (got ${subSize ? subSize[1] + subSize[2] : 'nothing'})`);
ok(topSize && subSize && Number(subSize[1]) <= Number(topSize[1]),
  'sub-entries are no larger than top-level entries');
ok(subSize && Number(subSize[1]) >= 0.7,
  `sub-entries stay legible (${subSize ? subSize[1] : '?'}rem)`);

// ---- 6. the swap --------------------------------------------------------
const swapObs = observers.find((o) => o.targets.has(nav));
ok(Boolean(swapObs), 'an observer watches the inline contents');
const scrollTo = (past) => swapObs.cb([
  { isIntersecting: !past, boundingClientRect: { top: past ? -500 : 100 }, target: nav },
]);

ok(panel.hidden === true && authorCard.hidden === false,
  'at the top: author card shown, sidebar contents hidden');
scrollTo(true);
ok(authorCard.hidden === true, 'scrolled past: author card hidden');
ok(panel.hidden === false, 'scrolled past: sidebar contents shown');
ok(sidebar._cls.has('toc-mode'), 'scrolled past: sidebar carries .toc-mode');
scrollTo(false);
ok(authorCard.hidden === false && panel.hidden === true, 'scrolled back up: restored');

// narrow screens never swap
mqWide = false;
scrollTo(true);
ok(authorCard.hidden === false && panel.hidden === true,
  'below the breakpoint the swap does not fire');
mqWide = true;

// ---- 7. scroll-spy lights BOTH copies -----------------------------------
const spy = observers.find((o) => o !== swapObs && o.targets.has(headEls[0]));
ok(Boolean(spy), 'an observer watches the headings');
spy.cb([{ isIntersecting: true, target: headEls[0] }]);
const lit = root.all.filter((e) => e.tagName === 'A' && e._cls.has('toc-on'));
ok(lit.length === 2, `the current section is lit in both copies (got ${lit.length})`);
spy.cb([{ isIntersecting: true, target: headEls[2] }]);
const lit2 = root.all.filter((e) => e.tagName === 'A' && e._cls.has('toc-on'));
ok(lit2.length === 2 && lit2.every((a) => a.textContent === 'Building a Forecast'),
  'moving on clears the previous highlight in both copies');

console.log(fail ? `\n${fail} FAILED` : '\nOK');
process.exit(fail ? 1 : 0);
