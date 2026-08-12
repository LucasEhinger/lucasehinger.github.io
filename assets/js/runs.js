/* Running page: mileage over time, plus the repeated routes ("Lucas Loops")
   grouped by scripts/strava_runs.py. Activity rows come from
   files/strava/activities.json, the same file /hikes reads. */
(function () {
  "use strict";

  var ACTIVITIES_URL = "/files/strava/activities.json";
  var RUNS_URL = "/files/strava/runs.json";

  var RUN_SPORTS = { Run: true, VirtualRun: true, TrailRun: true };

  // The "Lucas Loops only" filter. Deliberately a distance band and not loop
  // membership: the habit is an 11-mile run, and a run in the band belongs to
  // the story whether or not strava_runs.py matched it to a repeated route.
  var LOOP_BAND = { center: 11, tolerance: 1 };

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // Same validated categorical theme as the hikes page, so a year reads the
  // same colour on both. Worst adjacent CVD dE 9.1 light / 8.4 dark, worst
  // adjacent normal-vision dE 19.6 / 19.3 on this theme's own surfaces
  // (#ffffff / #474747). Several slots sit under 3:1 against the surface, so
  // identity never rests on colour alone — legends, hover readouts and the
  // monthly table all carry it too.
  var YEAR_COLORS = {
    light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
            "#008300", "#4a3aa7", "#e34948"],
    dark:  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
            "#008300", "#9085e9", "#e66767"]
  };

  // Direction is a two-way split, so it takes the first two categorical slots
  // (dE 25-34 apart, comfortably clear of the floor). Every chip states the
  // direction in words next to the swatch — the colour is never the only cue.
  var DIRECTIONS = {
    ccw:      { label: "counterclockwise", glyph: "↺", slot: 0 },
    cw:       { label: "clockwise",        glyph: "↻", slot: 1 },
    forward:  { label: "usual way",        glyph: "→", slot: 0 },
    reverse:  { label: "reversed",         glyph: "←", slot: 1 },
    unknown:  { label: "direction unclear", glyph: "—", slot: null }
  };

  var state = {
    data: null,
    loops: null,
    runs: [],
    from: null,   // "YYYY-MM"
    to: null,
    loopsOnly: false,
    loopSort: "count",
    openLoop: null,
    allYears: []
  };

  /* ---------- helpers ---------- */

  function $(id) { return document.getElementById(id); }

  // Held rather than looked up: the panel is detached every time the loop
  // grid's innerHTML is replaced, after which getElementById can't find it.
  var loopPanel = null;

  function fmt(value, decimals) {
    return Number(value).toLocaleString("en-US", {
      minimumFractionDigits: decimals || 0,
      maximumFractionDigits: decimals || 0
    });
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function ym(dateStr) { return dateStr.slice(0, 7); }

  function paceText(secondsPerMile) {
    if (!secondsPerMile) return "—";
    var minutes = Math.floor(secondsPerMile / 60);
    var seconds = Math.round(secondsPerMile % 60);
    if (seconds === 60) { minutes += 1; seconds = 0; }
    return minutes + ":" + (seconds < 10 ? "0" : "") + seconds + "/mi";
  }

  function durationText(seconds) {
    var h = Math.floor(seconds / 3600);
    var m = Math.round((seconds % 3600) / 60);
    if (m === 60) { h += 1; m = 0; }
    return h + ":" + (m < 10 ? "0" : "") + m;
  }

  function prettyDate(dateStr) {
    return MONTHS[parseInt(dateStr.slice(5, 7), 10) - 1] + " " +
      parseInt(dateStr.slice(8, 10), 10) + ", " + dateStr.slice(0, 4);
  }

  function median(values) {
    if (!values.length) return 0;
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function isDarkMode() {
    var theme = document.documentElement.getAttribute("data-theme");
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return !!(window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function chartTextColor() {
    var val = getComputedStyle(document.documentElement)
      .getPropertyValue("--global-text-color");
    if (val && val.trim()) return val.trim();
    return isDarkMode() ? "#f2f5fa" : "#333";
  }

  function slotColor(slot) {
    if (slot == null) return isDarkMode() ? "#9a9a9a" : "#888";
    return YEAR_COLORS[isDarkMode() ? "dark" : "light"][slot];
  }

  // Colour belongs to the year, not to its position in the filtered set, so
  // narrowing the date range never repaints the years that remain.
  function yearColor(year) {
    var slots = YEAR_COLORS[isDarkMode() ? "dark" : "light"];
    var index = state.allYears.indexOf(year);
    return slots[(index < 0 ? 0 : index) % slots.length];
  }

  function inRange(dateStr) {
    var key = ym(dateStr);
    return key >= state.from && key <= state.to;
  }

  function inBand(miles) {
    return !state.loopsOnly ||
      (miles >= LOOP_BAND.center - LOOP_BAND.tolerance &&
       miles <= LOOP_BAND.center + LOOP_BAND.tolerance);
  }

  // The one predicate every section scopes by, so the tiles, the charts and the
  // loop counts can never disagree about which runs are in play.
  function included(dateStr, miles) {
    return inRange(dateStr) && inBand(miles);
  }

  function bandText() {
    return (LOOP_BAND.center - LOOP_BAND.tolerance) + "–" +
      (LOOP_BAND.center + LOOP_BAND.tolerance) + " mi";
  }

  function filtered() {
    return state.runs.filter(function (r) {
      return included(r.date, r.distance_mi);
    });
  }

  /* ---------- controls ---------- */

  function buildMonthOptions() {
    var keys = state.runs.map(function (r) { return ym(r.date); });
    var min = keys.reduce(function (a, b) { return a < b ? a : b; });
    var max = keys.reduce(function (a, b) { return a > b ? a : b; });

    var options = [];
    var year = parseInt(min.slice(0, 4), 10);
    var month = parseInt(min.slice(5, 7), 10);
    while (true) {
      var key = year + "-" + (month < 10 ? "0" : "") + month;
      options.push(key);
      if (key === max) break;
      month += 1;
      if (month > 12) { month = 1; year += 1; }
    }

    ["rn-from", "rn-to"].forEach(function (id) {
      var select = $(id);
      select.innerHTML = options.map(function (key) {
        var label = MONTHS[parseInt(key.slice(5, 7), 10) - 1] + " " + key.slice(0, 4);
        return '<option value="' + key + '">' + label + "</option>";
      }).join("");
    });

    state.from = options[0];
    state.to = options[options.length - 1];
    $("rn-from").value = state.from;
    $("rn-to").value = state.to;
  }

  function applyPreset(preset) {
    var options = Array.prototype.map.call($("rn-from").options, function (o) {
      return o.value;
    });
    var last = options[options.length - 1];

    if (preset === "all") {
      state.from = options[0];
    } else if (preset === "ytd") {
      state.from = last.slice(0, 4) + "-01";
      if (options.indexOf(state.from) < 0) state.from = options[0];
    } else {
      var index = options.length - parseInt(preset, 10);
      state.from = options[index < 0 ? 0 : index];
    }
    state.to = last;

    $("rn-from").value = state.from;
    $("rn-to").value = state.to;

    document.querySelectorAll("#rn-presets button").forEach(function (b) {
      b.classList.toggle("active", b.dataset.preset === preset);
    });
    render();
  }

  function clearPresets() {
    document.querySelectorAll("#rn-presets button").forEach(function (b) {
      b.classList.remove("active");
    });
  }

  function wireControls() {
    $("rn-from").addEventListener("change", function () {
      state.from = this.value;
      if (state.from > state.to) { state.to = state.from; $("rn-to").value = state.to; }
      clearPresets();
      render();
    });

    $("rn-to").addEventListener("change", function () {
      state.to = this.value;
      if (state.to < state.from) { state.from = state.to; $("rn-from").value = state.from; }
      clearPresets();
      render();
    });

    document.querySelectorAll("#rn-presets button").forEach(function (button) {
      button.addEventListener("click", function () { applyPreset(button.dataset.preset); });
    });

    $("rn-loop-sort").addEventListener("change", function () {
      state.loopSort = this.value;
      renderLoops();
    });

    var loopsOnly = $("rn-loops-only");
    loopsOnly.addEventListener("click", function () {
      state.loopsOnly = !state.loopsOnly;
      loopsOnly.setAttribute("aria-pressed", state.loopsOnly ? "true" : "false");
      // Opening a loop, then filtering it away, would leave a detail panel with
      // nothing behind it.
      state.openLoop = null;
      render();
    });

    var info = $("rn-loops-info");
    var infobox = $("rn-loops-infobox");
    info.addEventListener("click", function () {
      var open = info.getAttribute("aria-expanded") === "true";
      info.setAttribute("aria-expanded", open ? "false" : "true");
      infobox.hidden = open;
    });
  }

  /* ---------- summary ---------- */

  function renderSummary(runs) {
    var miles = runs.reduce(function (sum, r) { return sum + r.distance_mi; }, 0);
    var seconds = runs.reduce(function (sum, r) { return sum + r.moving_time_s; }, 0);
    var paces = runs.filter(function (r) {
      return r.distance_mi > 0 && r.moving_time_s > 0;
    }).map(function (r) { return r.moving_time_s / r.distance_mi; });

    var longest = runs.reduce(function (max, r) {
      return !max || r.distance_mi > max.distance_mi ? r : max;
    }, null);

    var onLoop = runs.filter(function (r) {
      return state.loops && state.loops.assign[String(r.id)];
    }).length;

    var tiles = [
      { value: fmt(runs.length), label: "runs" },
      { value: fmt(miles, 0), label: "miles" },
      { value: fmt(seconds / 3600, 0), label: "hours moving" },
      { value: paceText(median(paces)), label: "median pace" },
      {
        value: longest ? fmt(longest.distance_mi, 1) : "—",
        label: longest ? "longest run (" + prettyDate(longest.date) + ")" : "longest run"
      },
      {
        value: runs.length ? fmt(100 * onLoop / runs.length, 0) + "%" : "—",
        label: "on a known loop"
      }
    ];

    $("rn-summary").innerHTML = tiles.map(function (tile) {
      return '<div class="rn-tile"><div class="rn-tile-value">' + tile.value +
        '</div><div class="rn-tile-label">' + tile.label + "</div></div>";
    }).join("");
  }

  /* ---------- charts ---------- */

  function monthlyTotals(runs) {
    // Every month between the endpoints, so a month off training shows as a
    // genuine gap in the bars rather than being skipped over.
    var totals = {};
    runs.forEach(function (r) {
      totals[ym(r.date)] = (totals[ym(r.date)] || 0) + r.distance_mi;
    });

    var keys = [];
    var year = parseInt(state.from.slice(0, 4), 10);
    var month = parseInt(state.from.slice(5, 7), 10);
    while (true) {
      var key = year + "-" + (month < 10 ? "0" : "") + month;
      keys.push(key);
      if (key === state.to) break;
      month += 1;
      if (month > 12) { month = 1; year += 1; }
    }

    return keys.map(function (key) {
      return { key: key, miles: totals[key] || 0 };
    });
  }

  function drawCharts(runs) {
    var textColor = chartTextColor();

    // Grid and axis lines stay recessive so the series read first — Plotly's
    // dark default draws them brighter than the data.
    var gridColor = "rgba(128,128,128,0.28)";
    var axisBase = {
      tickfont: { color: textColor },
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      linecolor: gridColor
    };
    var layoutBase = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: textColor },
      // Room at the top for the legend, and at the bottom for tick labels plus
      // an axis title.
      margin: { l: 60, r: 20, t: 34, b: 60 },
      // Above the plot rather than below it. A legend under the x axis has to
      // share that space with the tick labels and the axis title, and on the
      // cumulative chart's five year-series it lands on top of both.
      legend: {
        font: { color: textColor },
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "left",
        x: 0
      },
      hovermode: "closest"
    };
    var config = { displayModeBar: false, responsive: true };

    // Chart 1 — monthly volume, with a trailing 3-month mean over it. Both are
    // miles, so they share the one axis.
    var months = monthlyTotals(runs);
    var rolling = months.map(function (_, i) {
      if (i < 2) return null;
      var window = months.slice(i - 2, i + 1);
      return window.reduce(function (s, m) { return s + m.miles; }, 0) / 3;
    });

    // A real date axis rather than one category per month: over four years of
    // history a categorical axis prints all 50-odd labels and runs them into
    // the legend, while a date axis thins them to readable year marks.
    var labels = months.map(function (m) { return m.key + "-01"; });

    var monthlyTraces = [
      {
        x: labels,
        y: months.map(function (m) { return Math.round(m.miles * 10) / 10; }),
        name: "Monthly miles",
        type: "bar",
        // Bars on a date axis default to a day wide; give them most of a month
        // and let the remainder be the 2px-equivalent gap between them.
        width: 24 * 3600 * 1000 * 24,
        marker: { color: slotColor(0) },
        hovertemplate: "%{y:,.1f} mi<extra></extra>"
      },
      {
        x: labels,
        y: rolling.map(function (v) { return v == null ? null : Math.round(v * 10) / 10; }),
        name: "3-month average",
        type: "scatter",
        mode: "lines",
        line: { width: 2, color: slotColor(1) },
        hovertemplate: "%{y:,.1f} mi/mo<extra></extra>"
      }
    ];

    Plotly.newPlot("rn-chart-monthly", monthlyTraces, Object.assign({}, layoutBase, {
      hovermode: "x unified",
      xaxis: Object.assign({}, axisBase, {
        type: "date", hoverformat: "%b %Y", automargin: true
      }),
      yaxis: Object.assign({}, axisBase, {
        title: { text: "Miles", font: { color: textColor } },
        rangemode: "tozero",
        automargin: true
      })
    }), config);

    // Chart 2 — cumulative miles against day of year, one line per year, which
    // is the shape that answers "am I ahead of last year".
    var byYear = {};
    runs.forEach(function (r) {
      var year = r.date.slice(0, 4);
      var start = Date.UTC(parseInt(year, 10), 0, 1);
      var day = Math.round((Date.parse(r.date + "T00:00:00Z") - start) / 86400000) + 1;
      byYear[year] = byYear[year] || [];
      byYear[year].push({ day: day, miles: r.distance_mi });
    });

    var years = Object.keys(byYear).sort();
    var cumulativeTraces = years.map(function (year) {
      var entries = byYear[year].sort(function (a, b) { return a.day - b.day; });
      var total = 0;
      var xs = [0];
      var ys = [0];
      entries.forEach(function (e) {
        total += e.miles;
        xs.push(e.day);
        ys.push(Math.round(total * 10) / 10);
      });
      return {
        x: xs,
        y: ys,
        name: year,
        type: "scatter",
        mode: "lines",
        line: { width: 2, shape: "hv", color: yearColor(year) },
        hovertemplate: "%{y:,.0f} mi by day %{x}<extra>" + year + "</extra>"
      };
    });

    Plotly.newPlot("rn-chart-cumulative", cumulativeTraces, Object.assign({}, layoutBase, {
      xaxis: Object.assign({}, axisBase, {
        title: { text: "Day of year", font: { color: textColor } },
        range: [0, 366],
        automargin: true
      }),
      yaxis: Object.assign({}, axisBase, {
        title: { text: "Cumulative miles", font: { color: textColor } },
        rangemode: "tozero",
        automargin: true
      })
    }), config);

    renderMonthlyTable(months);
  }

  // The table view the contrast warning on the palette obliges, and a way to
  // read exact figures off the bars.
  function renderMonthlyTable(months) {
    var byYear = {};
    months.forEach(function (m) {
      var year = m.key.slice(0, 4);
      var index = parseInt(m.key.slice(5, 7), 10) - 1;
      byYear[year] = byYear[year] || new Array(12).fill(null);
      byYear[year][index] = m.miles;
    });

    var years = Object.keys(byYear).sort();
    var header = "<thead><tr><th>Year</th>" +
      MONTHS.map(function (m) { return "<th>" + m + "</th>"; }).join("") +
      "<th>Total</th></tr></thead>";

    var body = years.map(function (year) {
      var row = byYear[year];
      var total = row.reduce(function (s, v) { return s + (v || 0); }, 0);
      return "<tr><td>" + year + "</td>" +
        row.map(function (v) {
          return '<td class="num">' + (v == null ? "" : fmt(v, 0)) + "</td>";
        }).join("") +
        '<td class="num"><b>' + fmt(total, 0) + "</b></td></tr>";
    }).join("");

    $("rn-monthly-table").innerHTML = header + "<tbody>" + body + "</tbody>";
  }

  /* ---------- Lucas Loops ---------- */

  // A loop's stats follow the date range, so the counts always match the runs
  // being charted above.
  function loopRows() {
    return state.loops.loops.map(function (loop) {
      var activities = loop.activities.filter(function (a) {
        return included(a.date, a.mi);
      });
      var counts = {};
      activities.forEach(function (a) { counts[a.dir] = (counts[a.dir] || 0) + 1; });
      var paces = activities.filter(function (a) { return a.mi > 0 && a.s > 0; })
        .map(function (a) { return a.s / a.mi; });

      return {
        loop: loop,
        activities: activities,
        counts: counts,
        miles: activities.reduce(function (s, a) { return s + a.mi; }, 0),
        medianPace: median(paces),
        bestPace: paces.length ? Math.min.apply(null, paces) : 0,
        last: activities.length ? activities[0].date : null
      };
    }).filter(function (row) {
      return row.activities.length > 0;
    });
  }

  function sortRows(rows) {
    if (state.loopSort === "recent") {
      rows.sort(function (a, b) { return a.last < b.last ? 1 : a.last > b.last ? -1 : 0; });
    } else if (state.loopSort === "distance") {
      rows.sort(function (a, b) { return b.loop.median_mi - a.loop.median_mi; });
    } else if (state.loopSort === "miles") {
      rows.sort(function (a, b) { return b.miles - a.miles; });
    } else {
      rows.sort(function (a, b) {
        return b.activities.length - a.activities.length ||
          b.loop.median_mi - a.loop.median_mi;
      });
    }
    return rows;
  }

  function shapeSvg(loop, className) {
    var shape = loop.shape;
    // The path is normalised into a 0..100 box on its long side; pad by the
    // stroke so the outline is never clipped at the edges.
    var pad = 4;
    var viewBox = [-pad, -pad, shape.w + pad * 2, shape.h + pad * 2].join(" ");
    return '<svg class="' + className + '" viewBox="' + viewBox +
      '" preserveAspectRatio="xMidYMid meet" aria-hidden="true">' +
      '<path d="' + shape.d + '"></path></svg>';
  }

  function directionChips(counts, oriented) {
    // Fixed order so the same direction sits in the same place on every card.
    var order = oriented ? ["ccw", "cw", "unknown"] : ["forward", "reverse", "unknown"];
    return order.filter(function (key) { return counts[key]; }).map(function (key) {
      var meta = DIRECTIONS[key];
      return '<span class="rn-dir"><span class="rn-dot" style="--dir: ' +
        slotColor(meta.slot) + '"></span>' + meta.glyph + " " +
        counts[key] + " " + meta.label + "</span>";
    }).join("");
  }

  function renderLoops() {
    var rows = sortRows(loopRows());
    var totalRuns = rows.reduce(function (s, r) { return s + r.activities.length; }, 0);

    // Deliberately not "run 3+ times" — that is the cutoff for making the list
    // at all, but any narrowing of the range or the band can leave a route
    // showing fewer runs than that, and the copy must not claim otherwise.
    $("rn-loop-summary").textContent = rows.length
      ? rows.length + " repeated routes, " + totalRuns + " runs between them" +
        (state.loopsOnly ? " within " + bandText() : "") +
        " — click a loop for the full list."
      : "No repeated routes match the current range" +
        (state.loopsOnly ? " and the " + bandText() + " filter." : ".");

    $("rn-loops").innerHTML = rows.map(function (row, index) {
      var loop = row.loop;
      var open = state.openLoop === loop.id;
      return '<button type="button" class="rn-loop' + (open ? " open" : "") +
        '" data-loop="' + loop.id + '" data-index="' + index +
        '" aria-expanded="' + (open ? "true" : "false") + '">' +
        shapeSvg(loop, "rn-shape") +
        '<span class="rn-loop-body">' +
          '<span class="rn-loop-count">&times;' + row.activities.length + "</span>" +
          '<span class="rn-loop-name">' + escapeHtml(loop.name) + "</span>" +
          '<span class="rn-loop-meta">' + fmt(loop.median_mi, 2) + " mi typical &middot; " +
            paceText(row.medianPace) + "</span>" +
          '<span class="rn-dirs">' + directionChips(row.counts, loop.oriented) + "</span>" +
        "</span></button>";
    }).join("");

    $("rn-loops").querySelectorAll(".rn-loop").forEach(function (button) {
      button.addEventListener("click", function () {
        var id = button.dataset.loop;
        state.openLoop = state.openLoop === id ? null : id;
        renderLoops();
      });
    });

    renderLoopDetail(rows);
  }

  // How many cards fit per row, so the detail panel can be spliced in at the
  // end of the row holding the open card rather than after the whole grid.
  function gridColumns(buttons) {
    if (!buttons.length) return 1;
    var firstTop = buttons[0].offsetTop;
    var columns = 0;
    for (var i = 0; i < buttons.length; i++) {
      if (buttons[i].offsetTop !== firstTop) break;
      columns += 1;
    }
    return columns || 1;
  }

  function renderLoopDetail(rows) {
    var grid = $("rn-loops");
    if (loopPanel && loopPanel.parentNode) loopPanel.parentNode.removeChild(loopPanel);

    var index = -1;
    rows.forEach(function (row, i) {
      if (row.loop.id === state.openLoop) index = i;
    });
    if (index < 0) {
      loopPanel.style.display = "none";
      grid.appendChild(loopPanel);
      return;
    }

    var row = rows[index];
    var loop = row.loop;
    var buttons = grid.querySelectorAll(".rn-loop");
    var columns = gridColumns(buttons);
    var insertAt = Math.min((Math.floor(index / columns) + 1) * columns, buttons.length);

    var orientationNote = loop.oriented
      ? "Direction is clockwise or counterclockwise as seen on the map."
      : "This route doubles back on itself, so clockwise has no meaning on it — " +
        "directions are counted relative to the way it is usually run.";

    var head = '<div class="rn-detail-head">' +
      "<b>" + escapeHtml(loop.name) + "</b>" +
      "<span>" + row.activities.length + " runs</span>" +
      "<span>" + fmt(row.miles, 0) + " mi total</span>" +
      "<span>" + fmt(loop.min_mi, 2) + "–" + fmt(loop.max_mi, 2) + " mi</span>" +
      "<span>" + fmt(loop.median_gain_ft) + " ft gain</span>" +
      "<span>median " + paceText(row.medianPace) + "</span>" +
      "<span>best " + paceText(row.bestPace) + "</span>" +
      '<span class="rn-note">first ' + prettyDate(loop.first) + "</span>" +
      "</div>" +
      '<p class="rn-note" style="margin: 6px 0 0;">' + orientationNote +
      (row.activities.length > 12
        ? " All " + row.activities.length + " runs are listed below, newest first — scroll the list."
        : "") +
      "</p>";

    var items = row.activities.map(function (a) {
      var meta = DIRECTIONS[a.dir];
      return "<li>" +
        '<span class="rn-dot" style="--dir: ' + slotColor(meta.slot) + '" title="' +
          meta.label + '"></span>' +
        "<a href=\"https://www.strava.com/activities/" + a.id +
          '" rel="noopener">' + prettyDate(a.date) + "</a>" +
        '<span class="rn-li-meta">' + fmt(a.mi, 2) + " mi &middot; " +
          durationText(a.s) + " &middot; " + paceText(a.s / a.mi) + "</span>" +
        "</li>";
    }).join("");

    loopPanel.innerHTML = head +
      '<div class="rn-detail-cols">' +
        shapeSvg(loop, "rn-detail-map") +
        "<ul>" + items + "</ul>" +
      "</div>";
    loopPanel.style.display = "";

    if (insertAt >= buttons.length) {
      grid.appendChild(loopPanel);
    } else {
      grid.insertBefore(loopPanel, buttons[insertAt]);
    }
  }

  /* ---------- render ---------- */

  function render() {
    var runs = filtered();
    renderSummary(runs);
    drawCharts(runs);
    if (state.loops) renderLoops();
  }

  function init(data, loops) {
    state.data = data;
    state.loops = loops;
    state.runs = data.activities.filter(function (a) { return RUN_SPORTS[a.sport]; });

    state.allYears = Object.keys(state.runs.reduce(function (acc, r) {
      acc[r.date.slice(0, 4)] = true;
      return acc;
    }, {})).sort();

    loopPanel = $("rn-loop-detail");

    buildMonthOptions();
    wireControls();

    $("rn-updated").textContent = "Updated " + data.updated;

    if (loops) {
      $("rn-loop-method").innerHTML =
        "Two runs count as the same route when their tracks, snapped to an " +
        loops.params.cell_m + " m grid, overlap by at least " +
        Math.round(loops.params.threshold * 100) + "% " +
        "(intersection over union). Comparing sets of cells rather than " +
        "sequences of points makes the match indifferent to which direction " +
        "the loop was run and where it was started, so a loop and its reverse " +
        "are one route here — the direction is then counted separately " +
        "against the route's usual heading. " +
        loops.distinct_routes + " distinct routes came out of " +
        loops.tracked_runs + " runs with GPS; the " + loops.params.min_runs +
        "-run cutoff is what keeps one-offs off this list.";
    } else {
      $("rn-loop-method").textContent = "Loop data is unavailable right now.";
      $("rn-loops").innerHTML = "";
    }

    $("rn-loading").style.display = "none";
    $("rn-content").style.display = "";

    applyPreset("all");
  }

  function start() {
    Promise.all([
      fetch(ACTIVITIES_URL).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      }),
      // Loops are optional: mileage over time still works without them.
      fetch(RUNS_URL).then(function (r) {
        return r.ok ? r.json() : null;
      }).catch(function () { return null; })
    ]).then(function (results) {
      init(results[0], results[1]);
    }).catch(function (err) {
      $("rn-loading").textContent = "Could not load run data: " + err.message;
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  // The grid's column count changes with the viewport, which moves where the
  // open detail row belongs.
  var resizeTimer = null;
  window.addEventListener("resize", function () {
    if (!state.loops || !state.openLoop) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderLoops, 150);
  });

  // Charts and swatches are both theme-dependent, so a theme flip has to
  // repaint them.
  if (window.matchMedia) {
    var media = window.matchMedia("(prefers-color-scheme: dark)");
    if (media.addEventListener) {
      media.addEventListener("change", function () { if (state.data) render(); });
    }
  }
  new MutationObserver(function () {
    if (state.data) render();
  }).observe(document.documentElement, {
    attributes: true, attributeFilter: ["data-theme"]
  });
})();
