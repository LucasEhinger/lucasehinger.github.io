/* White Mountains trail coverage map. Data comes from files/strava/, written by
   scripts/wm_trails.py (network) and scripts/wm_coverage.py (scoring). */
(function () {
  "use strict";

  var COVERAGE_URL = "/files/strava/wm_coverage.json";
  var TRAILS_URL = "/files/strava/wm_trails.geojson";
  var PEAKS_URL = "/files/strava/peaks.json";

  var LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
  var LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";

  // Basemaps chosen to stay quiet under the trail lines: the trails are the
  // data, the terrain is context. CARTO's plain raster styles come in a matched
  // light/dark pair, so flipping the site theme doesn't change the map's
  // character, only its surface.
  var BASEMAPS = {
    light: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
  };
  var BASEMAP_ATTRIB =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> ' +
    'contributors, &copy; <a href="https://carto.com/attributions">CARTO</a>';

  // Four steps, one per visit count, plus the grey for ground never walked.
  // That grey is not a step of the ramp — it's the denominator, and it has to
  // stay legible: what's left is as interesting as what's done, and a grey that
  // fades into the basemap makes the network look smaller than it is.
  //
  // Both ramps are single-hue with monotone lightness, each stepped against its
  // own surface rather than flipped from the other — the dark ramp runs
  // dim-to-bright so even one visit clears the dark basemap.
  //
  // Green rather than the usual heatmap warm ramp: red is already spoken for by
  // selection, and a yellow-to-red heat scale would make a busy trail and a
  // selected one hard to tell apart at a glance.
  var RAMPS = {
    light: { none: "#8a929b", heat: ["#74c476", "#41ab5d", "#238b45", "#005a32"] },
    dark: { none: "#8b939c", heat: ["#3d9e5f", "#5ec77c", "#8fe0a0", "#c9f5cd"] }
  };
  var HEAT_MAX = 4;

  // The selected trail goes red. Red against a green ramp is the one pairing
  // red/green colour blindness cannot resolve, so selection never rests on the
  // hue: the line also doubles in weight, and clicking opens a panel naming the
  // trail. The colour is the fast signal for everyone else.
  var SELECTED = { light: "#c1272d", dark: "#ff6f5e" };

  // Summits get their own hue rather than a step of the trail ramp — they're a
  // different kind of thing, not more-or-less of the same thing. Amber sits
  // clear of both the green ramp and the red selection under every form of
  // colour blindness, and the markers carry a shape the trails don't have.
  var PEAK_COLOR = { light: "#a15c00", dark: "#f0b429" };

  var COMPLETE_PCT = 95;

  var state = {
    coverage: null,
    geojson: null,
    peaks: null,
    map: null,
    layer: null,
    peakLayer: null,
    boundary: null,
    region: "all",
    selected: null,
    showPeaks: true,
    showBoundaries: true,
    loading: false
  };

  function $(id) { return document.getElementById(id); }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function fmt(n, digits) {
    return n.toLocaleString(undefined, {
      minimumFractionDigits: digits, maximumFractionDigits: digits
    });
  }

  function isDarkMode() {
    var theme = document.documentElement.getAttribute("data-theme");
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return !!(window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function ramp() { return RAMPS[isDarkMode() ? "dark" : "light"]; }

  // Shade is now how many separate hikes crossed *this stretch* — nothing to do
  // with how complete its trail is. A trail walked end to end once and a trail
  // whose first mile you've done four times read differently, which is the
  // point: this shows where the boots actually went.
  //
  // The touched gate stays. Crossing Huntington Ravine Trail at a junction picks
  // up ~47 m of it, which is real ground but not a hike of that trail; without
  // this, every junction speckles the map with 1-visit stubs on trails the
  // detail panel correctly reports as never hiked.
  function colorFor(trail, visits) {
    var r = ramp();
    if (!visits || !trail.touched) return r.none;
    return r.heat[Math.min(visits, HEAT_MAX) - 1];
  }

  /* ---- data ---- */

  function loadScript(url) {
    return new Promise(function (resolve, reject) {
      var el = document.createElement("script");
      el.src = url;
      el.onload = resolve;
      el.onerror = function () { reject(new Error("failed to load " + url)); };
      document.head.appendChild(el);
    });
  }

  function loadCss(url) {
    var el = document.createElement("link");
    el.rel = "stylesheet";
    el.href = url;
    document.head.appendChild(el);
  }

  function loadJson(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + " -> " + r.status);
      return r.json();
    });
  }

  /* ---- rendering ---- */

  function trailFor(feature) {
    return state.coverage.trails[feature.properties.t];
  }

  function inRegion(feature) {
    if (state.region === "all") return true;
    // A trail that crosses a boundary belongs to both sides, so this is a
    // membership test rather than an equality one.
    return trailFor(feature).regions.indexOf(state.region) >= 0;
  }

  // The drawn boundary is the same ring the scoring uses, not a decorative
  // outline — if they could drift apart, the map would start lying about which
  // trails belong to what.
  //
  // With one region selected only its ring is drawn. On "all" every ring is
  // drawn and labelled, which is the only view that shows how the forest is
  // actually carved up — and where a line looks wrong.
  function drawBoundary() {
    if (state.boundary) {
      state.map.removeLayer(state.boundary);
      state.boundary = null;
    }
    if (!state.showBoundaries || !state.coverage.regions.length) return;

    var ink = isDarkMode() ? "#e8e2d0" : "#3a3a3a";
    var showingAll = state.region === "all";
    state.boundary = L.layerGroup();

    state.coverage.regions.forEach(function (row) {
      if (!row.ring) return;
      if (!showingAll && row.name !== state.region) return;

      var ring = L.polygon(row.ring, {
        color: ink,
        weight: showingAll ? 1 : 1.5,
        opacity: showingAll ? 0.45 : 0.7,
        dashArray: "6,5",
        fill: false,
        interactive: false
      });
      state.boundary.addLayer(ring);

      if (showingAll) {
        // A divIcon marker rather than a tooltip: permanent tooltips on
        // non-interactive paths came out shifted a long way west of the ring
        // they label, tracking latitude but not longitude. A marker positions
        // straight from its LatLng, and the CSS centres it on that point.
        state.boundary.addLayer(
          L.marker(L.latLngBounds(row.ring).getCenter(), {
            interactive: false,
            keyboard: false,
            icon: L.divIcon({
              className: "hm-region-label",
              // Inner span because Leaflet writes its own transform onto the
              // icon element inline, which would beat any centring rule here.
              html: "<span>" + escapeHtml(row.name) + "</span>"
            })
          })
        );
      }
    });

    state.boundary.addTo(state.map);
  }

  function isWalked(feature) {
    return feature.properties.n > 0 && trailFor(feature).touched;
  }

  // Out-of-region trails stay drawn, and stay clickable — Leaflet only honours
  // `interactive` when the layer is created, so a style pass could never have
  // switched it off anyway. They're dimmed enough to read as context rather
  // than content, but not so far that a trail filed in a neighbouring range
  // becomes an invisible thing you can still click.
  var DIM_OPACITY = 0.22;

  function styleFor(feature) {
    var visible = inRegion(feature);
    var walked = isWalked(feature);
    return {
      color: colorFor(trailFor(feature), walked ? feature.properties.n : 0),
      // Busier ground draws a touch heavier, so the heat reads even where the
      // shades are hard to separate against a dark tile.
      weight: walked ? 2 + 0.35 * Math.min(feature.properties.n, HEAT_MAX) : 1.6,
      opacity: visible ? (walked ? 0.95 : 0.8) : DIM_OPACITY,
      // setStyle merges, so this has to be cleared explicitly or a previously
      // selected trail keeps the dashes after it is deselected.
      dashArray: null
    };
  }

  // A stretch you've walked should sit on top of one you haven't where the two
  // meet, or the grey wins and the map reads emptier than it is.
  function raiseWalked() {
    state.layer.eachLayer(function (l) {
      if (isWalked(l.feature)) l.bringToFront();
    });
  }

  function restyle() {
    if (!state.layer) return;
    state.layer.setStyle(styleFor);
    raiseWalked();
    if (state.selected) highlight(state.selected);
  }

  // The whole selected trail goes red so you can see where it runs end to end,
  // but the stretches you've walked stay heavier and solid — otherwise
  // selecting a trail would hide the very split the rest of the map exists to
  // show.
  function highlight(name) {
    var color = SELECTED[isDarkMode() ? "dark" : "light"];
    state.layer.eachLayer(function (l) {
      if (trailFor(l.feature).name !== name) return;
      var walked = isWalked(l.feature);
      l.setStyle({
        color: color,
        weight: walked ? 5 : 2.5,
        opacity: walked ? 1 : 0.55,
        dashArray: walked ? null : "4,4"
      });
      l.bringToFront();
    });
  }

  function statsFor(region) {
    if (region === "all") return state.coverage.totals;
    var found = null;
    state.coverage.regions.forEach(function (r) {
      if (r.name === region) found = r;
    });
    return found;
  }

  function renderStats() {
    var s = statsFor(state.region);
    if (!s) return;
    var hiked = s.trails_hiked;
    var complete = s.trails_complete;
    $("hm-stats").innerHTML = [
      ["% of trails walked", fmt(s.pct, 1) + "%"],
      ["Miles walked", fmt(s.covered_miles, 1) + " / " + fmt(s.miles, 1)],
      ["Trails hiked", hiked + (s.trails ? " of " + s.trails : "")],
      ["Trails finished", String(complete)]
    ].map(function (pair) {
      return '<div class="hm-stat"><div class="hm-stat-value">' + pair[1] +
        '</div><div class="hm-stat-label">' + pair[0] + "</div></div>";
    }).join("");
  }

  function fitToRegion() {
    // With a region selected, fit the drawn boundary rather than the member
    // trails: membership is per trail, so a trail straddling two ranges brings
    // its whole length into the bounds and pulls the view well outside the
    // polygon the user just asked to look at.
    if (state.region !== "all") {
      var row = statsFor(state.region);
      if (row && row.ring) {
        state.map.fitBounds(L.latLngBounds(row.ring), { padding: [20, 20] });
        return;
      }
    }

    var bounds = null;
    state.layer.eachLayer(function (l) {
      bounds = bounds ? bounds.extend(l.getBounds()) : L.latLngBounds(l.getBounds());
    });
    if (bounds) state.map.fitBounds(bounds, { padding: [20, 20] });
  }

  function renderDetail(name) {
    var panel = $("hm-detail");
    if (!name) {
      panel.style.display = "none";
      panel.innerHTML = "";
      return;
    }

    var trail = null;
    state.coverage.trails.forEach(function (t) {
      if (t.name === name) trail = t;
    });
    if (!trail) return;

    panel.style.display = "";
    var head = '<div class="hm-detail-head"><b>' + escapeHtml(trail.name) + "</b>" +
      '<span class="hm-meta">' + fmt(trail.pct, 1) + "% of " + fmt(trail.miles, 2) + " mi</span>" +
      (trail.regions.length ? '<span class="hm-meta">' +
        escapeHtml(trail.regions.join(" · ")) + "</span>" : "") +
      '<button type="button" class="hm-close" id="hm-close">clear</button></div>';

    var body;
    if (!trail.hikes.length) {
      body = '<p class="hm-meta">No hikes have covered this trail yet.</p>';
    } else {
      body = "<ul>" + trail.hikes.map(function (h) {
        return "<li><span>" + h.date + "</span>" +
          '<a href="https://www.strava.com/activities/' + h.id +
          '" target="_blank" rel="noopener">' + escapeHtml(h.name) + "</a>" +
          '<span class="hm-meta">' + fmt(h.miles, 2) + " mi</span></li>";
      }).join("") + "</ul>";
    }

    panel.innerHTML = head + body;
    $("hm-close").addEventListener("click", function () {
      state.selected = null;
      renderDetail(null);
      restyle();
    });
  }

  function renderPeakDetail(peak) {
    var panel = $("hm-detail");
    panel.style.display = "";
    panel.innerHTML = '<div class="hm-detail-head"><b>' + escapeHtml(peak.name) + "</b>" +
      (peak.elev_ft ? '<span class="hm-meta">' + fmt(peak.elev_ft, 0) + " ft</span>" : "") +
      '<span class="hm-meta">' + peak.count +
      (peak.count === 1 ? " ascent" : " ascents") + "</span>" +
      '<button type="button" class="hm-close" id="hm-close">clear</button></div>' +
      "<ul>" + peak.ascents.map(function (a) {
        return "<li><span>" + a.date + "</span>" +
          '<a href="https://www.strava.com/activities/' + a.id +
          '" target="_blank" rel="noopener">' + escapeHtml(a.name) + "</a></li>";
      }).join("") + "</ul>";

    $("hm-close").addEventListener("click", function () {
      state.selected = null;
      renderDetail(null);
      restyle();
    });
  }

  // Only summits that sit within the trail network's own bounds — peaks.json
  // covers everywhere this athlete has been, and Mount Rainier has no business
  // on a White Mountains map.
  function buildPeaks() {
    var bounds = state.layer.getBounds();
    var color = PEAK_COLOR[isDarkMode() ? "dark" : "light"];

    state.peakLayer = L.layerGroup();
    state.peaks.peaks.forEach(function (peak) {
      if (!bounds.contains([peak.lat, peak.lon])) return;

      var marker = L.circleMarker([peak.lat, peak.lon], {
        radius: peak.count > 1 ? 5 : 4,
        color: "#ffffff",
        weight: 1,
        fillColor: color,
        fillOpacity: 0.95
      });
      marker.bindTooltip(
        peak.name + (peak.elev_ft ? " — " + fmt(peak.elev_ft, 0) + " ft" : "") +
          " (x" + peak.count + ")",
        { direction: "top" }
      );
      marker.on("click", function (e) {
        // Otherwise the click falls through to whatever trail is underneath and
        // the panel immediately shows that instead.
        L.DomEvent.stopPropagation(e);
        state.selected = null;
        restyle();
        renderPeakDetail(peak);
      });
      state.peakLayer.addLayer(marker);
    });

    if (state.showPeaks) state.peakLayer.addTo(state.map);
  }

  function restylePeaks() {
    if (!state.peakLayer) return;
    var color = PEAK_COLOR[isDarkMode() ? "dark" : "light"];
    state.peakLayer.eachLayer(function (m) { m.setStyle({ fillColor: color }); });
  }

  function renderLegend() {
    var r = ramp();
    // Only walked stretches are coloured, and the shade counts separate hikes
    // over that stretch — not how finished its trail is.
    var r = ramp();
    var keys = [[r.none, "Not hiked"]];
    r.heat.forEach(function (color, i) {
      keys.push([color, (i + 1) + (i + 1 === HEAT_MAX ? "+ hikes" : " hike" + (i ? "s" : ""))]);
    });
    $("hm-legend").innerHTML = keys.map(function (pair) {
      return '<span class="hm-key"><i style="background:' + pair[0] + '"></i>' +
        pair[1] + "</span>";
    }).join("") +
      '<span class="hm-key"><i class="hm-dot" style="background:' +
      PEAK_COLOR[isDarkMode() ? "dark" : "light"] + '"></i>Summit</span>';
  }

  function renderRegions() {
    var select = $("hm-region");
    var options = ['<option value="all">All of the WMNF</option>'];
    state.coverage.regions.forEach(function (r) {
      options.push('<option value="' + escapeHtml(r.name) + '">' +
        escapeHtml(r.name) + " (" + fmt(r.pct, 0) + "%)</option>");
    });
    select.innerHTML = options.join("");
    select.addEventListener("change", function () {
      state.region = select.value;
      state.selected = null;
      renderDetail(null);
      restyle();
      renderStats();
      drawBoundary();
      fitToRegion();
    });
  }

  function wirePeakToggle() {
    var box = $("hm-peaks-toggle");
    box.checked = state.showPeaks;
    box.addEventListener("change", function () {
      state.showPeaks = box.checked;
      if (!state.peakLayer) return;
      if (state.showPeaks) {
        state.peakLayer.addTo(state.map);
      } else {
        state.map.removeLayer(state.peakLayer);
      }
    });
  }

  function wireBoundaryToggle() {
    var box = $("hm-bounds-toggle");
    box.checked = state.showBoundaries;
    box.addEventListener("change", function () {
      state.showBoundaries = box.checked;
      drawBoundary();
    });
  }

  function buildMap() {
    loadCss(LEAFLET_CSS);

    state.map = L.map("hm-map", { scrollWheelZoom: false });
    state.tiles = L.tileLayer(BASEMAPS[isDarkMode() ? "dark" : "light"], {
      attribution: BASEMAP_ATTRIB, maxZoom: 17
    }).addTo(state.map);

    state.layer = L.geoJSON(state.geojson, {
      style: styleFor,
      onEachFeature: function (feature, layer) {
        var trail = trailFor(feature);
        // The real count, not the capped one the shade uses — a stretch you've
        // done seven times should say so.
        var n = feature.properties.n;
        var walked = trail.touched && n;
        layer.bindTooltip(
          trail.name + " — " + (walked
            ? n + (n === 1 ? " time" : " times") + ", " +
              fmt(trail.pct, 0) + "% of trail hiked"
            : "not hiked"),
          { sticky: true }
        );
        layer.on("click", function () {
          state.selected = trail.name;
          renderDetail(trail.name);
          restyle();
        });
      }
    }).addTo(state.map);

    raiseWalked();
    buildPeaks();
    drawBoundary();
    fitToRegion();

    // Scroll-wheel zoom is off so the page still scrolls past the map; clicking
    // into it is an explicit "I want to drive this now".
    state.map.on("click", function () { state.map.scrollWheelZoom.enable(); });
    state.map.on("mouseout", function () { state.map.scrollWheelZoom.disable(); });
  }

  function init() {
    if (state.loading) return;
    state.loading = true;

    Promise.all([
      loadJson(COVERAGE_URL),
      loadJson(TRAILS_URL),
      loadJson(PEAKS_URL),
      loadScript(LEAFLET_JS)
    ]).then(function (results) {
      state.coverage = results[0];
      state.geojson = results[1];
      state.peaks = results[2];

      $("hm-loading").style.display = "none";
      $("hm-content").style.display = "";

      renderRegions();
      wirePeakToggle();
      wireBoundaryToggle();
      renderStats();
      renderLegend();
      buildMap();

      $("hm-updated").textContent = "Updated " + state.coverage.updated;
    }).catch(function (err) {
      $("hm-loading").textContent =
        "Could not load the trail map (" + err.message + ").";
    });
  }

  // The GeoJSON is ~540 KB, which is not worth spending on a visitor who never
  // scrolls this far. Everything above renders without it.
  function watchForVisibility() {
    var section = $("hm-section");
    if (!section) return;

    if (!("IntersectionObserver" in window)) {
      init();
      return;
    }
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          observer.disconnect();
          init();
        }
      });
    }, { rootMargin: "300px" });
    observer.observe(section);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", watchForVisibility);
  } else {
    watchForVisibility();
  }

  // Theme flip: the basemap and the ramp both have to move, or dark trail lines
  // end up on a dark basemap.
  new MutationObserver(function () {
    if (!state.map) return;
    state.tiles.setUrl(BASEMAPS[isDarkMode() ? "dark" : "light"]);
    restyle();
    restylePeaks();
    drawBoundary();
    renderLegend();
  }).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"]
  });
})();
