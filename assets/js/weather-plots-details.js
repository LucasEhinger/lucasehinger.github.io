/* Lightweight plotter for /weather/details
   Loads a selected example JSON from /files/weather/examples and renders the same
   plots as the main weather page, excluding ML prediction plots. */

const d_defaultColors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
const d_modelMarkers = { hrrr: "circle", nam: "square", gfs: "diamond", rap: "triangle-up", ecmwf: "cross", nbm: "x" };
// hrrr is solid (no entry); nam/gfs match the dash/dot they use in the
// hand-built traces on the other plots.
const d_modelDash = { nam: "dash", gfs: "dot", rap: "dashdot", ecmwf: "longdash", nbm: "longdashdot" };
const D_METERS_TO_FEET = 3.28084;
const D_METERS_TO_MILES = 0.000621371;
const D_METERS_TO_FEMTO_PARSECS = 3.24078e-2;
const D_MM_TO_INCHES = 0.0393701;
// image rotation state
let d_imageIntervalId = null;
let d_currentImageIdx = 0;
let d_imageSrcs = [null, null];
const D_IMAGE_SWITCH_MS = 5000;

function d_precipToUnits(valueMm, units) {
  if (valueMm == null) return null;
  if (units === "imperial") return valueMm * D_MM_TO_INCHES;
  if (units === "stupid") return valueMm * D_METERS_TO_FEMTO_PARSECS * 0.001;
  return valueMm;
}
function d_precipLabel(units){ if(units==='imperial') return 'in'; if(units==='stupid') return 'fempto-pc'; return 'mm'; }
function d_precipRateLabel(units){ if(units==='imperial') return 'in/hr'; if(units==='stupid') return 'fempto-pc/hr'; return 'mm/hr'; }
function d_visibilityToUnits(valueMeters,units){ if(valueMeters==null) return null; if(units==='imperial') return valueMeters*D_METERS_TO_MILES; if(units==='stupid') return valueMeters*D_METERS_TO_FEMTO_PARSECS; return valueMeters; }
function d_visibilityLabel(units){ if(units==='imperial') return 'mi'; if(units==='stupid') return 'fempto-pc'; return 'm'; }
function d_heightToUnits(valueMeters,units){ if(valueMeters==null) return null; if(units==='imperial') return valueMeters*D_METERS_TO_FEET; if(units==='stupid') return valueMeters*D_METERS_TO_FEMTO_PARSECS; return valueMeters; }
function d_heightLabel(units){ if(units==='imperial') return 'ft'; if(units==='stupid') return 'fempto-pc'; return 'm'; }
function d_tempLabel(units){ if(units==='imperial') return 'F'; if(units==='stupid') return '°R'; return 'C'; }

// NBM reports "no ceiling" as a sentinel near 88888 m rather than a null, which
// would otherwise plot as an 88 km spike and flatten every real value.
const D_NBM_NO_CEILING = 88000;
function d_maskCeiling(v){ return (v==null || v>=D_NBM_NO_CEILING) ? null : v; }

// NBM probabilistic ceiling/visibility ladders. The thresholds are defined in
// feet/miles upstream, so label them in whichever unit the reader picked.
// Mt. Washington's summit is 1917 m, so the 2012 m rung is effectively
// "probability the cloud deck is below the summit" -- i.e. undercast.
const D_SUMMIT_M = 1917;
const D_CEIL_PROB = [
  { key:'ceil_prob_below_152m_nbm',  m:152,  ft:500 },
  { key:'ceil_prob_below_305m_nbm',  m:305,  ft:1000 },
  { key:'ceil_prob_below_610m_nbm',  m:610,  ft:2000 },
  { key:'ceil_prob_below_914m_nbm',  m:914,  ft:3000 },
  { key:'ceil_prob_below_2012m_nbm', m:2012, ft:6600 },
];
const D_VIS_PROB = [
  { key:'vis_prob_below_1609m_nbm', m:1609, mi:1 },
  { key:'vis_prob_below_3219m_nbm', m:3219, mi:2 },
  { key:'vis_prob_below_4828m_nbm', m:4828, mi:3 },
  { key:'vis_prob_below_8047m_nbm', m:8047, mi:5 },
];
// A mixing height of 0 m is physically impossible, so an all-zero series means
// the archive never populated the field (true for NBM on most 2024/25 example
// days). Drop it rather than draw a flat line that reads as a real measurement.
// Deliberately not applied to fields where zero is meaningful, e.g. 0% cloud.
function d_dropAllZero(trace){ if(!trace||!trace.y) return null; const nn=trace.y.filter(v=>v!=null); return (nn.length && nn.every(v=>Number(v)===0)) ? null : trace; }

function d_ceilProbLabel(rung, units){ const base = units==='imperial' ? `below ${rung.ft.toLocaleString()} ft` : `below ${rung.m.toLocaleString()} m`; return rung.m > D_SUMMIT_M ? `${base}<br>(≈ summit)` : base; }
function d_visProbLabel(rung, units){ return units==='imperial' ? `below ${rung.mi} mi` : `below ${rung.m.toLocaleString()} m`; }

function d_getSelectedModel(){ const selected = Array.from(document.querySelectorAll('input[name="model-toggle"]:checked')).map(el=>el.value); return { hrrr: selected.includes('hrrr'), nam: selected.includes('nam'), gfs: selected.includes('gfs'), rap: selected.includes('rap'), ecmwf: selected.includes('ecmwf'), nbm: selected.includes('nbm') }; }
function d_getSelectedUnits(){ return document.querySelector('input[name="units-toggle"]:checked')?.value || 'metric'; }

// Tight x-axis bounds so a plot only spans the range where it actually has
// plotted data. x values are ISO-like date strings that sort lexicographically.
function d_computeXRange(traces){ let lo=null, hi=null; (traces||[]).forEach(t=>{ if(!t||!t.x||!t.y) return; for(let i=0;i<t.x.length;i++){ if(t.y[i]==null||t.x[i]==null) continue; const xv=t.x[i]; if(lo===null||xv<lo) lo=xv; if(hi===null||xv>hi) hi=xv; } }); return (lo!==null&&hi!==null)?[lo,hi]:undefined; }

function d_convertTemp(kelvin, units){ if(kelvin==null) return null; if(units==='imperial') return (((kelvin-273.15)*9)/5+32).toFixed(2); if(units==='stupid') return ((kelvin*9)/5).toFixed(2); return (kelvin-273.15).toFixed(2); }

function d_approxAltitude(meters, units){ const unitLabel=d_heightLabel(units); const sigRound=(val,sig=2)=>{ if(!isFinite(val)||val===0) return val; const power=sig-Math.ceil(Math.log10(Math.abs(val))); const factor=Math.pow(10,power); return Math.round(val*factor)/factor; }; const raw = units==='imperial'?meters*D_METERS_TO_FEET:meters; const value = units==='imperial'?sigRound(raw,2):Math.round(raw); return `${value}${unitLabel}`; }
function d_levelLabel(pressure, approxMeters, model, units){ return `${pressure}mb (~${d_approxAltitude(approxMeters,units)})<br>(${model})`; }

function d_convertTimeToDateTime(timeValues, dateStr) {
  const baseParts = dateStr.split(" ");
  const dateOnly = baseParts[0];
  const timeOnly = baseParts[1];
  const [year,month,day] = dateOnly.split("-").map(Number);
  const [hour,minute] = timeOnly.split(":").map(Number);
  const baseDate = new Date(Date.UTC(year,month-1,day,hour,minute,0));
  return timeValues.map((time)=>{
    const offsetDate = new Date(baseDate.getTime() + time*60*60*1000);
    const easternTime = offsetDate.toLocaleString('en-US',{ timeZone: 'America/New_York', year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false });
    const parts = easternTime.split(', ');
    const [m,d,y] = parts[0].split('/');
    const [h,min] = parts[1].split(':');
    return `${y}-${m}-${d} ${h}:${min}`;
  });
}

function attachSimpleTooltips() {
  // Small info icons similar to main site; simplified
  const map = { plot1: 'Cloud Coverage (%)', plot2: 'Cloud ceiling/base', plot3: 'Temperatures', plot4: 'Boundary layer / mixing height', plot5: 'Relative humidity', plot6: '0°C isotherm', plot7: 'Visibility', plot11: 'NBM ceiling probability', plot12: 'NBM visibility probability', plot10: 'Precipitation' };
  Object.keys(map).forEach((id)=>{
    const el = document.getElementById(id); if(!el) return; el.style.position='relative';
    const old = el.querySelector('.plot-info-button'); if(old) old.remove();
    const btn = document.createElement('div'); btn.className='plot-info-button'; btn.textContent='ⓘ'; btn.style.position='absolute'; btn.style.top='8px'; btn.style.left='8px'; btn.style.zIndex='10'; el.appendChild(btn);
    const tip = document.createElement('div'); tip.className='plot-info-tooltip'; tip.textContent=map[id]; tip.style.position='absolute'; tip.style.top='30px'; tip.style.left='8px'; tip.style.display='none'; el.appendChild(tip);
    btn.addEventListener('mouseenter',()=>tip.style.display='block'); btn.addEventListener('mouseleave',()=>tip.style.display='none'); tip.addEventListener('mouseenter',()=>tip.style.display='block'); tip.addEventListener('mouseleave',()=>tip.style.display='none');
  });
}

function loadDetailsPlotsFromData(data){
  const selectedModel = d_getSelectedModel();
  const selectedUnits = d_getSelectedUnits();
  const showHRRR = !!selectedModel.hrrr; const showNAM = !!selectedModel.nam; const showGFS = !!selectedModel.gfs; const showRAP = !!selectedModel.rap; const showECMWF = !!selectedModel.ecmwf; const showNBM = !!selectedModel.nbm;
  const textColor = (getComputedStyle(document.documentElement).getPropertyValue('--text-color')||'#000').trim();
  const dateStr = data.date_str || '2025-01-01 00:00';
  let convertedDates;
  try{ convertedDates = d_convertTimeToDateTime(data.low_cloud_layer_percent_hrrr.x, dateStr); }catch(e){ console.error(e); convertedDates = data.low_cloud_layer_percent_hrrr.x; }
  // Guarded extra-model trace builder: returns null when the snapshot lacks the
  // field (e.g. older example days without RAP, or ECMWF/NBM which the example
  // archive doesn't reach), so a toggle never breaks a plot.
  // ECMWF/IFS runs 3-hourly against this 2-hourly axis, so it only lands on every
  // 6th hour with nulls between. Without connectgaps its lines would never join
  // and it would render as isolated markers; the gaps are cadence, not missing data.
  const d_extraTrace = (key, name, color, model, mapFn, extra) => { const d = data[key]; if(!d||!d.y) return null; return Object.assign({ x:convertedDates, y: mapFn ? d.y.map(mapFn) : d.y, mode:'lines+markers', type:'scatter', connectgaps: model==='ecmwf', name, line:{dash:d_modelDash[model],color}, marker:{symbol:d_modelMarkers[model]} }, extra||{}); };
  const d_rapTrace = (key, name, color, mapFn, extra) => d_extraTrace(key, name, color, 'rap', mapFn, extra);
  const d_pushTruthy = (arr, items) => items.forEach(t=>t&&arr.push(t));
  // Render a plot, or hide its container when it has no traces for the current
  // selection, so irrelevant/empty plots collapse instead of showing blank axes.
  const d_renderOrHide = (id, traces, layout) => { const el = document.getElementById(id); if(!traces||traces.length===0){ if(el) el.style.display='none'; return; } if(el && el.style.display==='none') el.style.display=''; if(layout && layout.xaxis) layout.xaxis.range = d_computeXRange(traces); Plotly.newPlot(id, traces, layout).then(()=>{ const e=document.getElementById(id); if(e) Plotly.Plots.resize(e); }); };

  // Plot 1: Cloud Coverage
  const c1 = d_defaultColors;
  const traces1 = [];
  if(showHRRR){ traces1.push({ x:convertedDates, y:data.low_cloud_layer_percent_hrrr.y, mode:'lines+markers', type:'scatter', name:'Low (HRRR)', line:{color:c1[0]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.middle_cloud_layer_percent_hrrr.y, mode:'lines+markers', type:'scatter', name:'Middle (HRRR)', line:{color:c1[1]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.high_cloud_layer_percent_hrrr.y, mode:'lines+markers', type:'scatter', name:'High (HRRR)', line:{color:c1[2]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.boundary_layer_cloud_layer_hrrr.y, mode:'lines+markers', type:'scatter', name:'Boundary (HRRR)', line:{color:c1[3]}, marker:{symbol:d_modelMarkers.hrrr} }); }
  if(showNAM){ traces1.push({ x:convertedDates, y:data.low_cloud_layer_percent_nam.y, mode:'lines+markers', type:'scatter', name:'Low (NAM)', line:{dash:'dash',color:c1[0]}, marker:{symbol:d_modelMarkers.nam} }, { x:convertedDates, y:data.middle_cloud_layer_percent_nam.y, mode:'lines+markers', type:'scatter', name:'Middle (NAM)', line:{dash:'dash',color:c1[1]}, marker:{symbol:d_modelMarkers.nam} }, { x:convertedDates, y:data.high_cloud_layer_percent_nam.y, mode:'lines+markers', type:'scatter', name:'High (NAM)', line:{dash:'dash',color:c1[2]}, marker:{symbol:d_modelMarkers.nam} }, { x:convertedDates, y:data.boundary_layer_cloud_layer_nam.y, mode:'lines+markers', type:'scatter', name:'Boundary (NAM)', line:{dash:'dash',color:c1[3]}, marker:{symbol:d_modelMarkers.nam} }); }
  if(showGFS){ traces1.push({ x:convertedDates, y:data.low_cloud_layer_percent_gfs.y, mode:'lines+markers', type:'scatter', name:'Low (GFS)', line:{dash:'dot',color:c1[0]}, marker:{symbol:d_modelMarkers.gfs} }, { x:convertedDates, y:data.middle_cloud_layer_percent_gfs.y, mode:'lines+markers', type:'scatter', name:'Middle (GFS)', line:{dash:'dot',color:c1[1]}, marker:{symbol:d_modelMarkers.gfs} }, { x:convertedDates, y:data.high_cloud_layer_percent_gfs.y, mode:'lines+markers', type:'scatter', name:'High (GFS)', line:{dash:'dot',color:c1[2]}, marker:{symbol:d_modelMarkers.gfs} }, { x:convertedDates, y:data.boundary_layer_cloud_layer_gfs.y, mode:'lines+markers', type:'scatter', name:'Boundary (GFS)', line:{dash:'dot',color:c1[3]}, marker:{symbol:d_modelMarkers.gfs} }); }
  if(showRAP){ d_pushTruthy(traces1, [ d_rapTrace('low_cloud_layer_percent_rap','Low (RAP)',c1[0]), d_rapTrace('middle_cloud_layer_percent_rap','Middle (RAP)',c1[1]), d_rapTrace('high_cloud_layer_percent_rap','High (RAP)',c1[2]), d_rapTrace('boundary_layer_cloud_layer_rap','Boundary (RAP)',c1[3]) ]); }
  if(showNBM){ d_pushTruthy(traces1, [ d_extraTrace('tcdc_surface_nbm','Total Cloud (NBM)',c1[0],'nbm') ]); }
  d_renderOrHide('plot1', traces1, { title:{text:'Cloud Coverage Percentage', font:{color:textColor}}, xaxis:{title:''}, yaxis:{title:'Cloud Coverage (%)', tickfont:{color:textColor}}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 2: Cloud ceiling/base
  const convertHeight = (m)=>d_heightToUnits(m, selectedUnits);
  const c2 = d_defaultColors;
  const traces2 = [];
  if(showHRRR){ traces2.push({ x:convertedDates, y:data.cloud_ceiling_m_hrrr.y.map(convertHeight), mode:'lines+markers', type:'scatter', name:'Cloud<br>Ceiling (HRRR)', line:{color:c2[0]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.cloud_base_m_hrrr.y.map(convertHeight), mode:'lines+markers', type:'scatter', name:'Cloud<br>Base (HRRR)', line:{color:c2[1]}, marker:{symbol:d_modelMarkers.hrrr} }); }
  if(showNAM){ traces2.push({ x:convertedDates, y:data.cloud_ceiling_nam.y.map(convertHeight), mode:'lines+markers', type:'scatter', name:'Cloud<br>Ceiling (NAM)', line:{dash:'dash',color:c2[0]}, marker:{symbol:d_modelMarkers.nam} }); }
  if(showGFS){ traces2.push({ x:convertedDates, y:data.cloud_ceiling_gfs.y.map(convertHeight), mode:'lines+markers', type:'scatter', name:'Cloud<br>Ceiling (GFS)', line:{dash:'dot',color:c2[0]}, marker:{symbol:d_modelMarkers.gfs} }); }
  if(showRAP){ d_pushTruthy(traces2, [ d_rapTrace('cloud_ceiling_m_rap','Cloud<br>Ceiling (RAP)',c2[0],convertHeight) ]); }
  if(showNBM){ const maskedHeight=(v)=>convertHeight(d_maskCeiling(v)); d_pushTruthy(traces2, [ d_extraTrace('cloud_ceiling_m_nbm','Cloud<br>Ceiling (NBM)',c2[0],'nbm',maskedHeight), d_extraTrace('cloud_base_m_nbm','Cloud<br>Base (NBM)',c2[1],'nbm',maskedHeight) ]); }
  d_renderOrHide('plot2', traces2, { title:{text:'Cloud Ceiling and Base Height', font:{color:textColor}}, xaxis:{}, yaxis:{title:`Height (${d_heightLabel(selectedUnits)})`}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 3: Temperatures
  const tempUnitLabel = d_tempLabel(selectedUnits);
  const c3 = d_defaultColors; const traces3 = [];
  if(showHRRR){ traces3.push({ x:convertedDates, y:data.tmp_1000mb_hrrr.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('1000',100,'HRRR',selectedUnits), line:{color:c3[0]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.tmp_925mb_hrrr.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('925',750,'HRRR',selectedUnits), line:{color:c3[1]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.tmp_850mb_hrrr.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('850',1500,'HRRR',selectedUnits), line:{color:c3[2]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.tmp_700mb_hrrr.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('700',3000,'HRRR',selectedUnits), line:{color:c3[3]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.tmp_500mb_hrrr.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('500',5500,'HRRR',selectedUnits), line:{color:c3[4]}, marker:{symbol:d_modelMarkers.hrrr} }, { x:convertedDates, y:data.tmp_2m_hrrr.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:'2m (HRRR)', line:{color:c3[5]}, marker:{symbol:d_modelMarkers.hrrr} }); }
  if(showNAM){ traces3.push({ x:convertedDates, y:data.tmp_1000mb_nam.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('1000',100,'NAM',selectedUnits), line:{dash:'dash',color:c3[0]}, marker:{symbol:d_modelMarkers.nam} }, { x:convertedDates, y:data.tmp_925mb_nam.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('925',750,'NAM',selectedUnits), line:{dash:'dash',color:c3[1]}, marker:{symbol:d_modelMarkers.nam} }, { x:convertedDates, y:data.tmp_850mb_nam.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('850',1500,'NAM',selectedUnits), line:{dash:'dash',color:c3[2]}, marker:{symbol:d_modelMarkers.nam} }, { x:convertedDates, y:data.tmp_2m_nam.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:'2m (NAM)', line:{dash:'dash',color:c3[5]}, marker:{symbol:d_modelMarkers.nam} }); }
  if(showGFS){ traces3.push({ x:convertedDates, y:data.tmp_1000mb_gfs.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('1000',100,'GFS',selectedUnits), line:{dash:'dot',color:c3[0]}, marker:{symbol:d_modelMarkers.gfs} }, { x:convertedDates, y:data.tmp_925mb_gfs.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('925',750,'GFS',selectedUnits), line:{dash:'dot',color:c3[1]}, marker:{symbol:d_modelMarkers.gfs} }, { x:convertedDates, y:data.tmp_850mb_gfs.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:d_levelLabel('850',1500,'GFS',selectedUnits), line:{dash:'dot',color:c3[2]}, marker:{symbol:d_modelMarkers.gfs} }, { x:convertedDates, y:data.tmp_2m_gfs.y.map(v=>d_convertTemp(v,selectedUnits)), mode:'lines+markers', type:'scatter', name:'2m (GFS)', line:{dash:'dot',color:c3[5]}, marker:{symbol:d_modelMarkers.gfs} }); }
  if(showRAP){ const tC=(v)=>d_convertTemp(v,selectedUnits); d_pushTruthy(traces3, [ d_rapTrace('tmp_1000mb_rap',d_levelLabel('1000',100,'RAP',selectedUnits),c3[0],tC), d_rapTrace('tmp_925mb_rap',d_levelLabel('925',750,'RAP',selectedUnits),c3[1],tC), d_rapTrace('tmp_850mb_rap',d_levelLabel('850',1500,'RAP',selectedUnits),c3[2],tC), d_rapTrace('tmp_700mb_rap',d_levelLabel('700',3000,'RAP',selectedUnits),c3[3],tC), d_rapTrace('tmp_500mb_rap',d_levelLabel('500',5500,'RAP',selectedUnits),c3[4],tC), d_rapTrace('tmp_2m_rap','2m (RAP)',c3[5],tC) ]); }
  if(showECMWF){ const tC=(v)=>d_convertTemp(v,selectedUnits); d_pushTruthy(traces3, [ d_extraTrace('tmp_1000mb_ecmwf',d_levelLabel('1000',100,'ECMWF',selectedUnits),c3[0],'ecmwf',tC), d_extraTrace('tmp_925mb_ecmwf',d_levelLabel('925',750,'ECMWF',selectedUnits),c3[1],'ecmwf',tC), d_extraTrace('tmp_850mb_ecmwf',d_levelLabel('850',1500,'ECMWF',selectedUnits),c3[2],'ecmwf',tC), d_extraTrace('tmp_700mb_ecmwf',d_levelLabel('700',3000,'ECMWF',selectedUnits),c3[3],'ecmwf',tC), d_extraTrace('tmp_500mb_ecmwf',d_levelLabel('500',5500,'ECMWF',selectedUnits),c3[4],'ecmwf',tC), d_extraTrace('tmp_2m_ecmwf','2m (ECMWF)',c3[5],'ecmwf',tC) ]); }
  if(showNBM){ d_pushTruthy(traces3, [ d_extraTrace('tmp_2m_nbm','2m (NBM)',c3[5],'nbm',(v)=>d_convertTemp(v,selectedUnits)) ]); }
  d_renderOrHide('plot3', traces3, { title:{text:'Temperature at Various Isobars', font:{color:textColor}}, xaxis:{}, yaxis:{title:`Temperature (${tempUnitLabel})`}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 4: Boundary layer / mixing height. The inversion cap: a shallow mixing
  // layer under a warm nose is what traps the deck below the summit.
  const c4 = d_defaultColors; const traces4 = [];
  if(showHRRR){ d_pushTruthy(traces4, [ d_extraTrace('hpbl_surface_hrrr','HPBL (HRRR)',c4[0],'hrrr',convertHeight) ]); }
  if(showNAM){ d_pushTruthy(traces4, [ d_extraTrace('hpbl_surface_nam','HPBL (NAM)',c4[0],'nam',convertHeight) ]); }
  if(showGFS){ d_pushTruthy(traces4, [ d_extraTrace('hpbl_surface_gfs','HPBL (GFS)',c4[0],'gfs',convertHeight) ]); }
  if(showRAP){ d_pushTruthy(traces4, [ d_rapTrace('hpbl_surface_rap','HPBL (RAP)',c4[0],convertHeight) ]); }
  if(showNBM){ d_pushTruthy(traces4, [ d_dropAllZero(d_extraTrace('mixing_height_nbm','Mixing Height (NBM)',c4[1],'nbm',convertHeight)) ]); }
  d_renderOrHide('plot4', traces4, { title:{text:'Boundary Layer / Mixing Height', font:{color:textColor}}, xaxis:{}, yaxis:{title:`Height (${d_heightLabel(selectedUnits)})`, rangemode:'tozero'}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 5: Relative Humidity
  const c5 = d_defaultColors; const traces5 = [];
  if(showHRRR) traces5.push({ x:convertedDates, y:data.rh_2m_hrrr.y, mode:'lines+markers', type:'scatter', name:'2m RH (HRRR)', line:{color:c5[0]}, marker:{symbol:d_modelMarkers.hrrr} });
  if(showNAM) traces5.push({ x:convertedDates, y:data.rh_2m_nam.y, mode:'lines+markers', type:'scatter', name:'2m RH (NAM)', line:{dash:'dash',color:c5[0]}, marker:{symbol:d_modelMarkers.nam} }, { x:convertedDates, y:data.rh_925mb_nam.y, mode:'lines+markers', type:'scatter', name:'925mb RH (NAM)', line:{dash:'dash',color:c5[1]}, marker:{symbol:d_modelMarkers.nam} });
  if(showGFS) traces5.push({ x:convertedDates, y:data.rh_2m_gfs.y, mode:'lines+markers', type:'scatter', name:'2m RH (GFS)', line:{dash:'dot',color:c5[0]}, marker:{symbol:d_modelMarkers.gfs} }, { x:convertedDates, y:data.rh_925mb_gfs.y, mode:'lines+markers', type:'scatter', name:'925mb RH (GFS)', line:{dash:'dot',color:c5[1]}, marker:{symbol:d_modelMarkers.gfs} });
  if(showRAP){ d_pushTruthy(traces5, [ d_rapTrace('rh_2m_rap','2m RH (RAP)',c5[0]), d_rapTrace('rh_925mb_rap','925mb RH (RAP)',c5[1]) ]); }
  if(showECMWF){ d_pushTruthy(traces5, [ d_extraTrace('rh_1000mb_ecmwf','1000mb RH (ECMWF)',c5[0],'ecmwf'), d_extraTrace('rh_925mb_ecmwf','925mb RH (ECMWF)',c5[1],'ecmwf') ]); }
  if(showNBM){ d_pushTruthy(traces5, [ d_extraTrace('rh_2m_nbm','2m RH (NBM)',c5[0],'nbm') ]); }
  d_renderOrHide('plot5', traces5, { title:{text:'Relative Humidity', font:{color:textColor}}, xaxis:{}, yaxis:{title:'Relative Humidity (%)'}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 6: 0°C isotherm
  const c6 = d_defaultColors; const traces6 = [];
  if(showHRRR) traces6.push({ x:convertedDates, y:data.hgt_0C_iso_hrrr.y.map(convertHeight), mode:'lines+markers', type:'scatter', name:'0°C Isotherm (HRRR)', line:{color:c6[0]}, marker:{symbol:d_modelMarkers.hrrr} });
  if(showNAM) traces6.push({ x:convertedDates, y:data.hgt_0C_iso_nam.y.map(convertHeight), mode:'lines+markers', type:'scatter', name:'0°C Isotherm (NAM)', line:{dash:'dash',color:c6[0]}, marker:{symbol:d_modelMarkers.nam} });
  if(showGFS) traces6.push({ x:convertedDates, y:data.hgt_0C_iso_gfs.y.map(convertHeight), mode:'lines+markers', type:'scatter', name:'0°C Isotherm (GFS)', line:{dash:'dot',color:c6[0]}, marker:{symbol:d_modelMarkers.gfs} });
  if(showRAP){ d_pushTruthy(traces6, [ d_rapTrace('hgt_0C_iso_rap','0°C Isotherm (RAP)',c6[0],convertHeight) ]); }
  d_renderOrHide('plot6', traces6, { title:{text:'0°C Isotherm Height', font:{color:textColor}}, xaxis:{}, yaxis:{title:`Height (${d_heightLabel(selectedUnits)})`}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 7: Visibility
  const c7 = d_defaultColors; const traces7 = [];
  const convertVisibility = (m)=>d_visibilityToUnits(m, selectedUnits);
  if(showHRRR) traces7.push({ x:convertedDates, y:data.vis_surface_hrrr.y.map(convertVisibility), mode:'lines+markers', type:'scatter', name:'Surface Visibility (HRRR)', line:{color:c7[0]}, marker:{symbol:d_modelMarkers.hrrr} });
  if(showNAM) traces7.push({ x:convertedDates, y:data.vis_surface_nam.y.map(convertVisibility), mode:'lines+markers', type:'scatter', name:'Surface Visibility (NAM)', line:{dash:'dash',color:c7[0]}, marker:{symbol:d_modelMarkers.nam} });
  if(showGFS) traces7.push({ x:convertedDates, y:data.vis_surface_gfs.y.map(convertVisibility), mode:'lines+markers', type:'scatter', name:'Surface Visibility (GFS)', line:{dash:'dot',color:c7[0]}, marker:{symbol:d_modelMarkers.gfs} });
  if(showRAP){ d_pushTruthy(traces7, [ d_rapTrace('vis_surface_rap','Surface Visibility (RAP)',c7[0],convertVisibility) ]); }
  if(showNBM){ d_pushTruthy(traces7, [ d_extraTrace('vis_surface_nbm','Surface Visibility (NBM)',c7[0],'nbm',convertVisibility) ]); }
  d_renderOrHide('plot7', traces7, { title:{text:'Surface Visibility', font:{color:textColor}}, xaxis:{}, yaxis:{title:`Visibility (${d_visibilityLabel(selectedUnits)})`}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 11: NBM probabilistic ceiling. The rung above the summit height is as
  // close to a native "undercast probability" as any of these models produce.
  const c11 = d_defaultColors; const traces11 = [];
  // Every trace here is NBM, so the per-model dash would just add noise: use
  // solid lines and let color carry the threshold.
  if(showNBM){ d_pushTruthy(traces11, D_CEIL_PROB.map((rung,i)=>d_extraTrace(rung.key, d_ceilProbLabel(rung,selectedUnits), c11[i%c11.length], 'nbm', null, { line:{color:c11[i%c11.length]} }))); }
  d_renderOrHide('plot11', traces11, { title:{text:'NBM Ceiling Probability', font:{color:textColor}}, xaxis:{}, yaxis:{title:'Probability of ceiling below (%)', range:[0,100]}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 12: NBM probabilistic visibility.
  const c12 = d_defaultColors; const traces12 = [];
  if(showNBM){ d_pushTruthy(traces12, D_VIS_PROB.map((rung,i)=>d_extraTrace(rung.key, d_visProbLabel(rung,selectedUnits), c12[i%c12.length], 'nbm', null, { line:{color:c12[i%c12.length]} }))); }
  d_renderOrHide('plot12', traces12, { title:{text:'NBM Visibility Probability', font:{color:textColor}}, xaxis:{}, yaxis:{title:'Probability of visibility below (%)', range:[0,100]}, legend:{font:{color:textColor}}, showlegend:true });

  // Plot 10: Precipitation
  const c9 = d_defaultColors; const traces10 = [];
  const convertPrecip = (mm)=>d_precipToUnits(mm, selectedUnits);
  const convertPrecipRate = (mmPerSec)=>{ if(mmPerSec==null) return null; const mmPerHr = mmPerSec*3600; return d_precipToUnits(mmPerHr, selectedUnits); };
  if(showHRRR){ traces10.push({ x:convertedDates, y:data.apcp_surface_hrrr.y.map(convertPrecip), mode:'lines+markers', type:'scatter', name:'Accumulated Precip (HRRR)', line:{color:c9[0]}, marker:{symbol:d_modelMarkers.hrrr}, yaxis:'y' }, { x:convertedDates, y:data.prate_surface_hrrr.y.map(convertPrecipRate), mode:'lines+markers', type:'scatter', name:'Precip Rate (HRRR)', line:{color:c9[1]}, marker:{symbol:d_modelMarkers.hrrr}, yaxis:'y2' }); }
  if(showNAM){ traces10.push({ x:convertedDates, y:data.apcp_surface_nam.y.map(convertPrecip), mode:'lines+markers', type:'scatter', name:'Accumulated Precip (NAM)', line:{dash:'dash',color:c9[0]}, marker:{symbol:d_modelMarkers.nam}, yaxis:'y' }, { x:convertedDates, y:data.prate_surface_nam.y.map(convertPrecipRate), mode:'lines+markers', type:'scatter', name:'Precip Rate (NAM)', line:{dash:'dash',color:c9[1]}, marker:{symbol:d_modelMarkers.nam}, yaxis:'y2' }); }
  if(showGFS){ traces10.push({ x:convertedDates, y:data.apcp_surface_gfs.y.map(convertPrecip), mode:'lines+markers', type:'scatter', name:'Accumulated Precip (GFS)', line:{dash:'dot',color:c9[0]}, marker:{symbol:d_modelMarkers.gfs}, yaxis:'y' }, { x:convertedDates, y:data.prate_surface_gfs.y.map(convertPrecipRate), mode:'lines+markers', type:'scatter', name:'Precip Rate (GFS)', line:{dash:'dot',color:c9[1]}, marker:{symbol:d_modelMarkers.gfs}, yaxis:'y2' }); }
  if(showRAP){ d_pushTruthy(traces10, [ d_rapTrace('apcp_surface_rap','Accumulated Precip (RAP)',c9[0],convertPrecip,{yaxis:'y'}), d_rapTrace('prate_surface_rap','Precip Rate (RAP)',c9[1],convertPrecipRate,{yaxis:'y2'}) ]); }
  if(showNBM){ d_pushTruthy(traces10, [ d_extraTrace('apcp_surface_nbm','Accumulated Precip (NBM)',c9[0],'nbm',convertPrecip,{yaxis:'y'}) ]); }
  d_renderOrHide('plot10', traces10, { title:{text:'Precipitation', font:{color:textColor}}, xaxis:{}, yaxis:{title:`Accumulated (${d_precipLabel(selectedUnits)})`, rangemode:'tozero'}, yaxis2:{title:`Rate (${d_precipRateLabel(selectedUnits)})`, overlaying:'y', side:'right', rangemode:'tozero'}, legend:{font:{color:textColor}}, showlegend:true });

  // update last update display if available
  const lastEl = document.getElementById('last-update'); if(lastEl){ try{ const baseParts = (data.date_str||'').split(' '); const dateOnly=baseParts[0]; const timeOnly=baseParts[1]; const [y,m,d]=dateOnly.split('-').map(Number); const [hh,mm]=timeOnly.split(':').map(Number); const utcDate=new Date(Date.UTC(y,m-1,d,hh,mm,0)); const eastern = utcDate.toLocaleString('en-US',{timeZone:'America/New_York', year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false}); lastEl.textContent = `Model timestamp: ${eastern} ET`; }catch(e){ lastEl.textContent=''; } }

  // tooltips
  setTimeout(attachSimpleTooltips,100);
}

function tryPadDay(path){
  // if path ends with YYYY-MM-D.json where D is single-digit, pad day to two digits
  const m = path.match(/(.*_)(\d{4}-\d{2}-)(\d)\.json$/);
  if(m){ return `${m[1]}${m[2]}0${m[3]}.json`; }
  return null;
}

function tryUnpadDay(path){
  // if path ends with YYYY-MM-0D.json where D is single-digit, unpad to YYYY-MM-D.json
  const m = path.match(/(.*_)(\d{4}-\d{2}-)0(\d)\.json$/);
  if(m){ return `${m[1]}${m[2]}${m[3]}.json`; }
  return null;
}

function fetchJsonWithPad(path){
  // Try original, then padded, then unpadded variants to be robust to filename variants
  return fetch(path).then(r=>{
    if(r.ok) return r.json();
    const padded = tryPadDay(path);
    if(padded){ return fetch(padded).then(r2=>{ if(r2.ok) return r2.json(); const unp = tryUnpadDay(path); if(unp){ return fetch(unp).then(r3=>{ if(r3.ok) return r3.json(); throw new Error('Not found'); }); } throw new Error('Not found'); }); }
    const unp = tryUnpadDay(path);
    if(unp){ return fetch(unp).then(r3=>{ if(r3.ok) return r3.json(); throw new Error('Not found'); }); }
    throw new Error('Not found');
  });
}

function loadDetailsPlots(filePath){
  fetchJsonWithPad(filePath).then(data=>{
    loadDetailsPlotsFromData(data);
    try{
      // derive base date from filename and set Obs/Tower images
      const m = filePath.match(/([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})\.json$/);
      if(m){
        // pad month/day to two digits for image filenames
        const parts = m[1].split('-');
        const base = `${parts[0]}-${parts[1].padStart(2,'0')}-${parts[2].padStart(2,'0')}`;
        // prefer images in the same directory as the JSON; fall back to top-level examples
        const dir = filePath.replace(/\/[^\/]*$/, '');
        const obsPath = `${dir}/${base}_Obs.jpg`;
        const towerPath = `${dir}/${base}_Tower.jpg`;
          const imgEl = document.getElementById('example-image');
          const dotEls = document.querySelectorAll('#image-dots .dot');
          console.debug('Searching example images for', base, 'candidates in dir', dir);

          // Helper to test image existence
          const testImage = (src) => new Promise((res) => {
            const p = new Image();
            p.onload = () => res({ src, ok: true });
            p.onerror = () => res({ src, ok: false });
            p.src = src;
          });

          // Build ordered candidate lists: prefer the JSON's directory, then a per-date subfolder, then top-level
          const parentExamples = '/files/weather/examples';
          const subdir = `${parentExamples}/${base}`;
          // if dir points to the top-level examples folder (no per-date folder was part of the path), prefer the per-date subdir
          const effectiveDir = (dir === parentExamples || dir === parentExamples.replace(/\/$/,'') ) ? subdir : dir;
          const obsCandidates = Array.from(new Set([`${effectiveDir}/${base}_Obs.jpg`, `${subdir}/${base}_Obs.jpg`, `${parentExamples}/${base}_Obs.jpg`]));
          const towerCandidates = Array.from(new Set([`${effectiveDir}/${base}_Tower.jpg`, `${subdir}/${base}_Tower.jpg`, `${parentExamples}/${base}_Tower.jpg`]));

          // Test candidates in order, one at a time, stopping at the first hit.
          // (Testing in parallel would fire requests for every fallback path even
          // when the first candidate already exists, spamming 404s.)
          const findFirst = (candidates, idx = 0) => {
            if (idx >= candidates.length) return Promise.resolve(null);
            return testImage(candidates[idx]).then(r => r.ok ? candidates[idx] : findFirst(candidates, idx + 1));
          };

          Promise.all([findFirst(obsCandidates), findFirst(towerCandidates)]).then(([obsFound, towerFound])=>{
            console.debug('found', { obsFound, towerFound });
            const finalObs = obsFound || obsCandidates[0];
            const finalTower = towerFound || towerCandidates[0];
            // assemble image sources array; include only those that actually exist (or keep both as fallback)
            const exists = [];
            if(obsFound) exists.push(finalObs);
            if(towerFound && finalTower !== finalObs) exists.push(finalTower);
            if(exists.length===0){ d_imageSrcs = [finalObs, finalTower]; }
            else { d_imageSrcs = exists; }

            const prevBtn = document.getElementById('image-prev');
            const nextBtn = document.getElementById('image-next');

            // show a given image index, updating the img, dots, and restarting the rotation timer
            const showImageIdx = (idx, restartTimer) => {
              d_currentImageIdx = ((idx % d_imageSrcs.length) + d_imageSrcs.length) % d_imageSrcs.length;
              if(imgEl){ imgEl.src = d_imageSrcs[d_currentImageIdx]; imgEl.alt = `Example image ${base} ${d_currentImageIdx===0?'(Obs)':'(Tower)'}`; }
              dotEls.forEach(e=>{ const i=Number(e.dataset.index); e.classList.toggle('active', i===d_currentImageIdx); e.style.opacity = i===d_currentImageIdx? '0.9':'0.5'; });
              if(restartTimer && d_imageSrcs.length>1){
                if(d_imageIntervalId) clearInterval(d_imageIntervalId);
                d_imageIntervalId = setInterval(()=>showImageIdx(d_currentImageIdx+1, false), D_IMAGE_SWITCH_MS);
              }
            };

            showImageIdx(0, false);

            dotEls.forEach(el=>{
              const idx = Number(el.dataset.index);
              el.onclick = ()=>showImageIdx(idx, true);
            });
            if(prevBtn) prevBtn.onclick = ()=>showImageIdx(d_currentImageIdx-1, true);
            if(nextBtn) nextBtn.onclick = ()=>showImageIdx(d_currentImageIdx+1, true);

            // start rotation (only if we have more than one image)
            if(d_imageIntervalId) clearInterval(d_imageIntervalId);
            if(d_imageSrcs.length>1){
              d_imageIntervalId = setInterval(()=>showImageIdx(d_currentImageIdx+1, false), D_IMAGE_SWITCH_MS);
            } else {
              d_imageIntervalId = null;
            }
          }).catch(err=>{ console.warn('Error testing example images',err); d_imageSrcs = [obsPath, towerPath]; if(imgEl) imgEl.src = d_imageSrcs[0]; });
      }
    }catch(e){ console.warn('Could not set example image',e); }
  }).catch(e=>console.error('Error loading example file',e));
}

function populateDateSelect(){
  fetch('/files/weather/examples/index.json').then(r=>r.json()).then(list=>{
    const sel = document.getElementById('details-date-select'); sel.innerHTML='';
    list.forEach((it,idx)=>{ const opt = document.createElement('option'); opt.value = normalizeExamplePath(it.path); opt.textContent = it.label; sel.appendChild(opt); });
    // load first
    if(list.length>0) loadDetailsPlots(normalizeExamplePath(list[0].path));
  }).catch(e=>{ console.error('Error loading index.json',e); const sel=document.getElementById('details-date-select'); sel.innerHTML=''; const opt=document.createElement('option'); opt.textContent='No dates available'; sel.appendChild(opt); });
}

function normalizeExamplePath(path){
  if(!path) return path;
  // normalize any YYYY-M?-D? to YYYY-MM-DD
  const m = path.match(/(.*_)(\d{4})-(\d{1,2})-(\d{1,2})(\.json|_Obs\.jpg|_Tower\.jpg)$/);
  if(m){ const prefix=m[1], y=m[2], mo=m[3].padStart(2,'0'), d=m[4].padStart(2,'0'), suffix=m[5]; return `${prefix}${y}-${mo}-${d}${suffix}`; }
  return path;
}

// Plotly sizes to the container at plot time only, so the two-column grid would
// keep its old pixel width after a window resize or phone rotation.
let d_resizeTimer;
window.addEventListener('resize', ()=>{ clearTimeout(d_resizeTimer); d_resizeTimer = setTimeout(()=>{ ['plot1','plot2','plot3','plot4','plot5','plot6','plot7','plot11','plot12','plot10'].forEach((id)=>{ const el=document.getElementById(id); if(el && el.style.display!=='none' && el.data) Plotly.Plots.resize(el); }); }, 150); });

document.getElementById('details-date-select').addEventListener('change',(e)=>{ loadDetailsPlots(normalizeExamplePath(e.target.value)); });
document.querySelectorAll('input[name="model-toggle"]').forEach(inp=>inp.addEventListener('change',()=>{ const cur = normalizeExamplePath(document.getElementById('details-date-select').value); loadDetailsPlots(cur); }));
document.querySelectorAll('input[name="units-toggle"]').forEach(inp=>inp.addEventListener('change',()=>{ const cur = normalizeExamplePath(document.getElementById('details-date-select').value); loadDetailsPlots(cur); }));

populateDateSelect();
