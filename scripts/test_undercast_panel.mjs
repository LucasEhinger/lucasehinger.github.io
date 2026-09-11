// Render states for the headline undercast panel on /weather/.
//
// The panel has to distinguish three things that are easy to conflate and
// dangerous to get wrong: "an undercast is expected", "no undercast is
// expected", and "we do not know". The third is the one that matters -- a panel
// that quietly disappears reads as the second to anyone who does not know the
// panel exists. This pulls the two functions straight out of weather-plots.js,
// runs them against a stub DOM, and prints what each state renders.
//
//   node scripts/test_undercast_panel.mjs <positive-payload.json> <negative-payload.json>
//
// The payload files are whatever predict_current_model() returns; any saved
// predictions_all.json["current"] block will do.

import fs from 'fs';
const src = fs.readFileSync('assets/js/weather-plots.js','utf8');
const grab = (name) => {
  const i = src.indexOf(`function ${name}(`);
  if (i < 0) throw new Error('not found: '+name);
  let d=0, j=src.indexOf('{', i);
  for (let k=j;k<src.length;k++){ if(src[k]==='{')d++; else if(src[k]==='}'){d--; if(!d) return src.slice(i,k+1);} }
};
const code = grab('parseRunTimeUTC') + '\n' + grab('renderUndercastHeadline');

// minimal DOM
const mk = (id) => ({ id, hidden:false, textContent:'', innerHTML:'', style:{}, title:'',
  className:'', classList:{ toggle(c,on){ this._[c]=on; }, _:{} },
  children:[], appendChild(c){ this.children.push(c); if(c.textContent) this.textContent += c.textContent; } });
const nodes = {}; ['undercast-headline','uh-verdict','uh-when','uh-strip','uh-axis','uh-foot'].forEach(i=>nodes[i]=mk(i));
global.document = { getElementById:(i)=>nodes[i]||null,
  createElement:(t)=>mk('<'+t+'>'),
  createTextNode:(t)=>({ textContent:t, className:'' }) };

const fn = new Function('document', code + '; return { renderUndercastHeadline, parseRunTimeUTC };')(global.document);

const run = (label, payload, ds) => {
  Object.values(nodes).forEach(n=>{ n.textContent=''; n.innerHTML=''; n.children=[]; n.hidden=false; });
  fn.renderUndercastHeadline(payload, payload && payload.date_str, ds);
  const h = nodes['undercast-headline'];
  console.log(`\n== ${label}  (hidden=${h.hidden})`);
  if (h.hidden) return;
  console.log('   verdict:', nodes['uh-verdict'].textContent);
  console.log('   when   :', nodes['uh-when'].textContent);
  console.log('   bars   :', nodes['uh-strip'].children.length,
              'filled:', nodes['uh-strip'].children.filter(b=>/uh-hit/.test(b.className)).length);
  console.log('   axis   :', nodes['uh-axis'].children.map(c=>c.textContent).join('  ->  '));
  console.log('   foot   :', nodes['uh-foot'].textContent.slice(0,200));
};

const pos = JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const nowISO = (off) => { const d=new Date(Date.now()+off*3600*1000); return d.toISOString().slice(0,16).replace('T',' '); };
pos.date_str = nowISO(-1);
run('undercast expected (Mt Washington)', { current: pos, date_str: pos.date_str }, '1');
run('other summit', { current: pos, date_str: pos.date_str }, '2');

const neg = JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
neg.date_str = nowISO(-1);
run('no undercast', { current: neg, date_str: neg.date_str }, '1');
run('model did not run', { date_str: nowISO(-1) }, '1');
run('nulls beyond max lead', { current: { x:[0,2,4], y:[null,null,null], probability:[null,null,null], model:{} }, date_str: nowISO(-1) }, '1');
run('serving path failed', { current: { status:'unavailable', reason:'ECMWF is only 0% populated' }, date_str: nowISO(-1) }, '1');
