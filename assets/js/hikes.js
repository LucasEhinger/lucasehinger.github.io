/* Activity stats page: date range + category filtering, charts, and peaks.
   Data comes from files/strava/, written by scripts/strava_sync.py and
   scripts/strava_peaks.py. */
(function () {
  "use strict";

  var ACTIVITIES_URL = "/files/strava/activities.json";
  var PEAKS_URL = "/files/strava/peaks.json";

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // Display names and a stable colour per category, used by both the toggles
  // and the charts so a category reads the same everywhere on the page.
  var CATEGORY_META = {
    hiking:       { label: "Hiking",        color: "#2e7d32" },
    trailrunning: { label: "Trail running", color: "#7cb342" },
    running:      { label: "Running",       color: "#1f77b4" },
    biking:       { label: "Biking",        color: "#ef6c00" },
    backcountryski: { label: "Backcountry ski", color: "#5e35b1" },
    nordicski:    { label: "Nordic ski",    color: "#9575cd" },
    alpineski:    { label: "Resort ski",    color: "#c2185b" },
    water:        { label: "Water",         color: "#0097a7" }
  };

  // Categorical palette for the per-year lines, stepped separately for each
  // surface rather than flipped. Validated on this theme's own backgrounds
  // (#ffffff / #474747): worst adjacent CVD dE 9.1 light / 8.4 dark, worst
  // adjacent normal-vision dE 19.6 / 19.3. Several slots sit under 3:1 against
  // the surface, so identity never rests on colour alone — the legend, the
  // hover readout, and the activities table all carry it too.
  var YEAR_COLORS = {
    light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
            "#008300", "#4a3aa7", "#e34948"],
    dark:  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
            "#008300", "#9085e9", "#e66767"]
  };

  var METRICS = {
    distance: { label: "Distance", unit: "mi", field: "distance_mi", decimals: 1 },
    elevation: { label: "Elevation gain", unit: "ft", field: "gain_ft", decimals: 0 },
    time: { label: "Moving time", unit: "h", field: "hours", decimals: 1 }
  };

  var state = {
    data: null,
    peaks: null,
    sportCategory: {},
    selected: {},
    from: null,   // "YYYY-MM"
    to: null,
    metric: "distance",
    sort: { column: 0, descending: true },
    openPeak: null,
    peakList: "all",
    categoriesOpen: true,
    winterOnly: false,
    peakSort: "elevation",
    allYears: []
  };

  /* ---------- helpers ---------- */

  function $(id) { return document.getElementById(id); }

  // Held rather than looked up: the panel gets detached every time the peak
  // grid's innerHTML is replaced, at which point getElementById can't find it.
  var peakPanel = null;

  function fmt(value, decimals) {
    return Number(value).toLocaleString("en-US", {
      minimumFractionDigits: decimals || 0,
      maximumFractionDigits: decimals || 0
    });
  }

  function hoursOf(activity) { return activity.moving_time_s / 3600; }

  function ym(dateStr) { return dateStr.slice(0, 7); }

  function prettyMonth(key) {
    return MONTHS[parseInt(key.slice(5, 7), 10) - 1] + " " + key.slice(0, 4);
  }

  function durationText(seconds) {
    var h = Math.floor(seconds / 3600);
    var m = Math.round((seconds % 3600) / 60);
    if (m === 60) { h += 1; m = 0; }
    return h + ":" + (m < 10 ? "0" : "") + m;
  }

  function chartTextColor() {
    var val = getComputedStyle(document.documentElement)
      .getPropertyValue("--global-text-color");
    if (val && val.trim()) return val.trim();
    var theme = document.documentElement.getAttribute("data-theme");
    if (theme === "dark") return "#f2f5fa";
    if (theme === "light") return "#333";
    var dark = window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    return dark ? "#f2f5fa" : "#333";
  }

  function isDarkMode() {
    var theme = document.documentElement.getAttribute("data-theme");
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return !!(window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  // Colour belongs to the year, not to its position in the filtered set, so
  // narrowing the date range never repaints the years that remain.
  function yearColor(year) {
    var slots = YEAR_COLORS[isDarkMode() ? "dark" : "light"];
    var index = state.allYears.indexOf(year);
    return slots[(index < 0 ? 0 : index) % slots.length];
  }

  /* ---------- filtering ---------- */

  function activeCategories() {
    return Object.keys(state.selected).filter(function (c) {
      return state.selected[c];
    });
  }

  function inRange(activity) {
    var key = ym(activity.date);
    if (state.from && key < state.from) return false;
    if (state.to && key > state.to) return false;
    return true;
  }

  function filtered() {
    return state.data.activities.filter(function (a) {
      return inRange(a) && state.selected[state.sportCategory[a.sport]];
    });
  }

  /* ---------- controls ---------- */

  function buildMonthOptions() {
    var keys = state.data.activities.map(function (a) { return ym(a.date); });
    var min = keys.reduce(function (a, b) { return a < b ? a : b; });
    var max = keys.reduce(function (a, b) { return a > b ? a : b; });

    var options = [];
    var year = parseInt(min.slice(0, 4), 10);
    var month = parseInt(min.slice(5, 7), 10);
    while (true) {
      var key = year + "-" + (month < 10 ? "0" : "") + month;
      options.push(key);
      if (key >= max) break;
      month += 1;
      if (month > 12) { month = 1; year += 1; }
    }

    ["hk-from", "hk-to"].forEach(function (id) {
      var select = $(id);
      select.innerHTML = options.map(function (key) {
        return '<option value="' + key + '">' + prettyMonth(key) + "</option>";
      }).join("");
    });

    state.from = options[0];
    state.to = options[options.length - 1];
    $("hk-from").value = state.from;
    $("hk-to").value = state.to;
  }

  function buildCategoryToggles() {
    var container = $("hk-categories");
    var present = Object.keys(CATEGORY_META).filter(function (category) {
      return state.data.activities.some(function (a) {
        return state.sportCategory[a.sport] === category;
      });
    });

    container.innerHTML = present.map(function (category) {
      var meta = CATEGORY_META[category];
      return '<button type="button" class="hk-cat active" data-category="' +
        category + '" style="--cat:' + meta.color + '">' + meta.label + "</button>";
    }).join("");

    present.forEach(function (category) { state.selected[category] = true; });

    container.querySelectorAll(".hk-cat").forEach(function (button) {
      button.addEventListener("click", function () {
        var category = button.dataset.category;
        // Never let every category be switched off — an empty page reads as
        // broken rather than as a filter result.
        if (state.selected[category] && activeCategories().length === 1) return;
        state.selected[category] = !state.selected[category];
        button.classList.toggle("active", state.selected[category]);
        render();
      });
    });
  }

  function applyPreset(preset) {
    var options = Array.prototype.map.call($("hk-from").options, function (o) {
      return o.value;
    });
    var last = options[options.length - 1];
    var from = options[0];

    if (preset === "ytd") {
      from = last.slice(0, 4) + "-01";
    } else if (preset !== "all") {
      var months = parseInt(preset, 10);
      var index = Math.max(0, options.length - months);
      from = options[index];
    }

    state.from = from;
    state.to = last;
    $("hk-from").value = from;
    $("hk-to").value = last;

    document.querySelectorAll("#hk-presets button").forEach(function (b) {
      b.classList.toggle("active", b.dataset.preset === preset);
    });
    render();
  }

  function wireControls() {
    $("hk-from").addEventListener("change", function () {
      state.from = this.value;
      if (state.from > state.to) { state.to = state.from; $("hk-to").value = state.to; }
      clearPresets();
      render();
    });

    $("hk-to").addEventListener("change", function () {
      state.to = this.value;
      if (state.to < state.from) { state.from = state.to; $("hk-from").value = state.from; }
      clearPresets();
      render();
    });

    document.querySelectorAll("#hk-presets button").forEach(function (button) {
      button.addEventListener("click", function () { applyPreset(button.dataset.preset); });
    });

    $("hk-metric").addEventListener("change", function () {
      state.metric = this.value;
      drawCharts();
    });

    $("hk-winter").addEventListener("click", function () {
      state.winterOnly = !state.winterOnly;
      state.openPeak = null;
      renderPeaks();
    });

    $("hk-peak-sort").addEventListener("change", function () {
      state.peakSort = this.value;
      renderPeaks();
    });
  }

  function clearPresets() {
    document.querySelectorAll("#hk-presets button").forEach(function (b) {
      b.classList.remove("active");
    });
  }

  /* ---------- summary ---------- */

  function totalsFor(activities) {
    var totals = { count: activities.length, distance: 0, gain: 0, seconds: 0, high: null };
    activities.forEach(function (a) {
      totals.distance += a.distance_mi;
      totals.gain += a.gain_ft;
      totals.seconds += a.moving_time_s;
      if (a.high_point_ft != null && (totals.high == null || a.high_point_ft > totals.high)) {
        totals.high = a.high_point_ft;
      }
    });
    return totals;
  }

  function renderSummary(activities) {
    var t = totalsFor(activities);
    var tiles = [
      { label: "Distance", value: fmt(t.distance, 0), unit: "mi" },
      { label: "Vertical", value: fmt(t.gain, 0), unit: "ft" },
      { label: "Moving time", value: fmt(t.seconds / 3600, 0), unit: "hrs" },
      { label: "Activities", value: fmt(t.count, 0), unit: "" }
    ];

    $("hk-summary").innerHTML = tiles.map(function (tile) {
      return '<div class="hk-card"><h3>' + tile.label + "</h3>" +
        '<div class="big">' + tile.value +
        (tile.unit ? "<span> " + tile.unit + "</span>" : "") + "</div></div>";
    }).join("");
  }

  function renderCategoryCards(activities) {
    var cards = activeCategories().map(function (category) {
      var subset = activities.filter(function (a) {
        return state.sportCategory[a.sport] === category;
      });
      if (!subset.length) return "";

      var t = totalsFor(subset);
      var meta = CATEGORY_META[category];
      var longest = subset.reduce(function (max, a) {
        return a.distance_mi > max ? a.distance_mi : max;
      }, 0);

      var rows = [
        ["Activities", fmt(t.count, 0)],
        ["Moving time", fmt(t.seconds / 3600, 0) + " h"],
        ["Vertical", fmt(t.gain, 0) + " ft"],
        ["Longest", fmt(longest, 1) + " mi"]
      ];
      if (t.high != null) rows.push(["High point", fmt(t.high, 0) + " ft"]);

      return '<div class="hk-card" style="border-top: 3px solid ' + meta.color + '">' +
        "<h3>" + meta.label + "</h3>" +
        '<div class="big">' + fmt(t.distance, 0) + "<span> mi</span></div><ul>" +
        rows.map(function (r) {
          return "<li><span>" + r[0] + "</span> <b>" + r[1] + "</b></li>";
        }).join("") + "</ul></div>";
    });

    $("hk-category-cards").innerHTML = cards.join("");
  }

  /* ---------- table ---------- */

  function renderTable(activities) {
    var rows = activities.slice();
    var column = state.sort.column;
    var descending = state.sort.descending;

    var accessors = [
      function (a) { return a.date; },
      function (a) { return a.name.toLowerCase(); },
      function (a) { return a.sport; },
      function (a) { return a.distance_mi; },
      function (a) { return a.gain_ft; },
      function (a) { return a.moving_time_s; },
      function (a) { return a.high_point_ft == null ? -1 : a.high_point_ft; }
    ];

    rows.sort(function (a, b) {
      var x = accessors[column](a);
      var y = accessors[column](b);
      if (x < y) return descending ? 1 : -1;
      if (x > y) return descending ? -1 : 1;
      return 0;
    });

    $("hk-table-count").textContent = fmt(rows.length, 0) +
      (rows.length === 1 ? " activity" : " activities");

    $("hk-tbody").innerHTML = rows.map(function (a) {
      var meta = CATEGORY_META[state.sportCategory[a.sport]];
      return "<tr>" +
        "<td>" + a.date + "</td>" +
        '<td><a href="https://www.strava.com/activities/' + a.id +
        '" target="_blank" rel="noopener">' + escapeHtml(a.name) + "</a></td>" +
        '<td><span class="hk-dot" style="background:' + meta.color + '"></span>' +
        a.sport + "</td>" +
        '<td class="num">' + fmt(a.distance_mi, 1) + "</td>" +
        '<td class="num">' + fmt(a.gain_ft, 0) + "</td>" +
        '<td class="num">' + durationText(a.moving_time_s) + "</td>" +
        '<td class="num">' +
        (a.high_point_ft == null ? "&mdash;" : fmt(a.high_point_ft, 0)) + "</td>" +
        "</tr>";
    }).join("");

    document.querySelectorAll("#hk-table th").forEach(function (th, index) {
      th.classList.toggle("sorted-asc", index === column && !descending);
      th.classList.toggle("sorted-desc", index === column && descending);
    });
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function wireTableSorting() {
    document.querySelectorAll("#hk-table th").forEach(function (th, index) {
      th.addEventListener("click", function () {
        if (state.sort.column === index) {
          state.sort.descending = !state.sort.descending;
        } else {
          state.sort.column = index;
          state.sort.descending = true;
        }
        renderTable(filtered());
      });
    });
  }

  /* ---------- charts ---------- */

  function metricValue(activity) {
    var metric = METRICS[state.metric];
    return metric.field === "hours" ? hoursOf(activity) : activity[metric.field];
  }

  function drawCharts() {
    var activities = filtered();
    var metric = METRICS[state.metric];
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
      margin: { l: 60, r: 20, t: 30, b: 50 },
      legend: { font: { color: textColor }, orientation: "h", y: -0.2 },
      hovermode: "closest"
    };
    var config = { displayModeBar: false, responsive: true };

    // Chart 1 — month of year on the x-axis, one line per calendar year, so
    // seasonality and year-over-year change are both readable at a glance.
    var byYearMonth = {};
    activities.forEach(function (a) {
      var year = a.date.slice(0, 4);
      var monthIndex = parseInt(a.date.slice(5, 7), 10) - 1;
      byYearMonth[year] = byYearMonth[year] || new Array(12).fill(0);
      byYearMonth[year][monthIndex] += metricValue(a);
    });

    var years = Object.keys(byYearMonth).sort();

    // Months outside the data window get null, not zero: the first months of
    // 2022 and the rest of the current year are "no data", and drawing them as
    // zero would imply months off training that never happened.
    var keys = activities.map(function (a) { return ym(a.date); });
    var minKey = keys.length ? keys.reduce(function (a, b) { return a < b ? a : b; }) : null;
    var maxKey = keys.length ? keys.reduce(function (a, b) { return a > b ? a : b; }) : null;

    var monthlyTraces = years.map(function (year) {
      return {
        x: MONTHS,
        y: byYearMonth[year].map(function (v, monthIndex) {
          var key = year + "-" + (monthIndex < 9 ? "0" : "") + (monthIndex + 1);
          if (minKey && (key < minKey || key > maxKey)) return null;
          return Math.round(v * 10) / 10;
        }),
        name: year,
        type: "scatter",
        mode: "lines+markers",
        line: { width: 2, color: yearColor(year) },
        marker: { size: 8, color: yearColor(year) },
        hovertemplate: "%{y:,.1f} " + metric.unit + "<extra>" + year + "</extra>"
      };
    });

    Plotly.newPlot("hk-chart-monthly", monthlyTraces, Object.assign({}, layoutBase, {
      // Every year at once for the hovered month — the comparison the chart is
      // for, and it names each year in text rather than by colour alone.
      hovermode: "x unified",
      xaxis: Object.assign({}, axisBase),
      yaxis: Object.assign({}, axisBase, {
        title: { text: metric.label + " (" + metric.unit + ")", font: { color: textColor } },
        rangemode: "tozero",
        automargin: true
      })
    }), config);

    // Chart 2 — the same metric per year, split by category.
    var byCategoryYear = {};
    activities.forEach(function (a) {
      var category = state.sportCategory[a.sport];
      var year = a.date.slice(0, 4);
      byCategoryYear[category] = byCategoryYear[category] || {};
      byCategoryYear[category][year] = (byCategoryYear[category][year] || 0) + metricValue(a);
    });

    var yearlyTraces = activeCategories().filter(function (category) {
      return byCategoryYear[category];
    }).map(function (category) {
      return {
        x: years,
        y: years.map(function (y) {
          return Math.round((byCategoryYear[category][y] || 0) * 10) / 10;
        }),
        name: CATEGORY_META[category].label,
        type: "bar",
        marker: { color: CATEGORY_META[category].color },
        hovertemplate: "%{x} " + CATEGORY_META[category].label +
          "<br>%{y:,.1f} " + metric.unit + "<extra></extra>"
      };
    });

    Plotly.newPlot("hk-chart-yearly", yearlyTraces, Object.assign({}, layoutBase, {
      barmode: "stack",
      bargap: 0.35,
      xaxis: Object.assign({}, axisBase, { type: "category" }),
      yaxis: Object.assign({}, axisBase, {
        title: { text: metric.label + " (" + metric.unit + ")", font: { color: textColor } },
        automargin: true
      })
    }), config);
  }

  /* ---------- peaks ---------- */

  // Calendar winter, solstice to equinox — the peakbagging definition, not the
  // meteorological Dec/Jan/Feb one. Solstice and equinox drift by a day or so
  // year to year; Dec 21 and Mar 20 are the common convention.
  function isWinterDate(date) {
    var monthDay = date.slice(5);
    return monthDay >= "12-21" || monthDay <= "03-20";
  }

  function ascentsInRange(peak) {
    return peak.ascents.filter(function (a) {
      var key = ym(a.date);
      if (state.from && key < state.from) return false;
      if (state.to && key > state.to) return false;
      if (state.winterOnly && !isWinterDate(a.date)) return false;
      return true;
    });
  }

  // Peaks belonging to a curated list, keyed by canonical name. Names are
  // unique within a list, which is what lets Mount Adams (NH) and Mount Adams
  // (WA) stay distinct.
  function listEntries(listKey) {
    var byName = {};
    state.peaks.peaks.forEach(function (peak) {
      if (peak.lists && peak.lists.indexOf(listKey) !== -1) {
        byName[peak.name] = peak;
      }
    });
    return byName;
  }

  function renderPeakFilters() {
    var lists = state.peaks.lists || {};
    var chips = ['<button type="button" class="hk-plist' +
      (state.peakList === "all" ? " active" : "") + '" data-list="all">All peaks</button>'];

    Object.keys(lists).forEach(function (key) {
      var byName = listEntries(key);
      var climbed = lists[key].members.filter(function (m) {
        var peak = byName[m.name];
        return peak && ascentsInRange(peak).length > 0;
      }).length;

      chips.push('<button type="button" class="hk-plist' +
        (state.peakList === key ? " active" : "") + '" data-list="' + key + '">' +
        escapeHtml(lists[key].label) + ' <b>' + climbed + " / " + lists[key].total +
        "</b></button>");
    });

    $("hk-peak-filters").innerHTML = chips.join("");
    $("hk-peak-filters").querySelectorAll(".hk-plist").forEach(function (button) {
      button.addEventListener("click", function () {
        state.peakList = button.dataset.list;
        state.openPeak = null;
        renderPeaks();
      });
    });

    // Winter stacks on top of the list choice, so the chip counters above
    // already read as "NH48 climbed in winter" when it's on.
    var winter = $("hk-winter");
    winter.classList.toggle("active", state.winterOnly);
    winter.setAttribute("aria-pressed", state.winterOnly ? "true" : "false");
  }

  function sortPeakRows(rows) {
    var byName = function (a, b) {
      return a.peak.name.localeCompare(b.peak.name);
    };

    if (state.peakSort === "name") {
      rows.sort(byName);
    } else if (state.peakSort === "frequency") {
      rows.sort(function (a, b) {
        return b.ascents.length - a.ascents.length || byName(a, b);
      });
    } else {
      // Elevation, highest first. Peaks OSM has no elevation for sort last
      // rather than to the top as a zero would.
      rows.sort(function (a, b) {
        var x = a.peak.elev_ft == null ? -1 : a.peak.elev_ft;
        var y = b.peak.elev_ft == null ? -1 : b.peak.elev_ft;
        return y - x || byName(a, b);
      });
    }
  }

  function renderPeaks() {
    if (!state.peaks) return;
    renderPeakFilters();

    // Peaks come from hikes and trail runs only, and are scoped by the date
    // range rather than the category toggles.
    var rows;

    if (state.peakList === "all") {
      rows = state.peaks.peaks.map(function (peak) {
        return { peak: peak, ascents: ascentsInRange(peak) };
      }).filter(function (row) { return row.ascents.length > 0; });

      $("hk-peak-count").textContent = fmt(rows.length, 0) +
        (rows.length === 1 ? " named peak" : " named peaks") +
        (state.winterOnly ? " climbed in winter" : "");
    } else {
      // A list view is a checklist: every member shows, climbed or not.
      var spec = state.peaks.lists[state.peakList];
      var byName = listEntries(state.peakList);

      rows = spec.members.map(function (member) {
        var peak = byName[member.name];
        return {
          peak: peak || { name: member.name, lat: 0, elev_ft: member.elev_ft },
          ascents: peak ? ascentsInRange(peak) : []
        };
      });

      var climbed = rows.filter(function (r) { return r.ascents.length > 0; }).length;
      $("hk-peak-count").textContent = climbed + " of " + spec.total + " " +
        spec.label + " climbed" + (state.winterOnly ? " in winter" : "");
    }

    sortPeakRows(rows);

    if (!rows.length) {
      $("hk-peaks").innerHTML = '<p class="hk-note">No summits in this date range.</p>';
      peakPanel.style.display = "none";
      return;
    }

    $("hk-peaks").innerHTML = rows.map(function (row) {
      var peak = row.peak;
      var key = peak.name + "|" + peak.lat;
      var isOpen = state.openPeak === key;
      var done = row.ascents.length > 0;

      return '<button type="button" class="hk-peak' + (isOpen ? " open" : "") +
        (done ? "" : " unclimbed") + '" data-key="' + escapeHtml(key) + '"' +
        (done ? "" : " disabled") + ">" +
        '<span class="hk-peak-name">' + escapeHtml(peak.name) + "</span>" +
        '<span class="hk-peak-meta">' +
        (peak.elev_ft ? fmt(peak.elev_ft, 0) + " ft" : "") + "</span>" +
        '<span class="hk-peak-count">' +
        (done ? "&times;" + row.ascents.length : "&mdash;") + "</span>" +
        "</button>";
    }).join("");

    $("hk-peaks").querySelectorAll(".hk-peak").forEach(function (button) {
      button.addEventListener("click", function () {
        var key = button.dataset.key;
        state.openPeak = state.openPeak === key ? null : key;
        renderPeaks();
      });
    });

    renderPeakDetail(rows);
  }

  // How many cards the responsive grid is currently fitting per row. Measured
  // rather than assumed, since the column count comes from auto-fill.
  function gridColumns(buttons) {
    if (!buttons.length) return 1;
    var top = buttons[0].offsetTop;
    var count = 0;
    for (var i = 0; i < buttons.length; i++) {
      if (buttons[i].offsetTop !== top) break;
      count++;
    }
    return count || 1;
  }

  // The ascent list is spliced into the grid as a full-width row directly
  // beneath the row holding the clicked peak: adjacent to what was clicked,
  // without leaving a hole beside it the way an in-card expansion would.
  function renderPeakDetail(rows) {
    var panel = peakPanel;
    var openIndex = -1;
    rows.forEach(function (row, index) {
      if (state.openPeak === row.peak.name + "|" + row.peak.lat) openIndex = index;
    });

    if (openIndex < 0) {
      panel.innerHTML = "";
      panel.style.display = "none";
      return;
    }

    var match = rows[openIndex];
    var grid = $("hk-peaks");
    var buttons = Array.prototype.slice.call(grid.querySelectorAll(".hk-peak"));
    var columns = gridColumns(buttons);
    var rowEnd = Math.min(
      buttons.length - 1,
      Math.floor(openIndex / columns) * columns + columns - 1
    );

    grid.insertBefore(panel, buttons[rowEnd].nextSibling);

    var peak = match.peak;
    panel.style.display = "";
    panel.innerHTML = '<div class="hk-detail-head">' +
      "<b>" + escapeHtml(peak.name) + "</b>" +
      (peak.elev_ft ? '<span class="hk-peak-meta">' + fmt(peak.elev_ft, 0) + " ft</span>" : "") +
      '<span class="hk-peak-meta">' + match.ascents.length +
      (match.ascents.length === 1 ? " ascent" : " ascents") + "</span>" +
      '<a class="hk-peak-meta" href="https://www.openstreetmap.org/?mlat=' + peak.lat +
      "&mlon=" + peak.lon + "#map=14/" + peak.lat + "/" + peak.lon +
      '" target="_blank" rel="noopener">map</a>' +
      "</div><ul>" +
      match.ascents.map(function (a) {
        return "<li><span>" + a.date + "</span>" +
          '<a href="https://www.strava.com/activities/' + a.id +
          '" target="_blank" rel="noopener">' + escapeHtml(a.name) + "</a></li>";
      }).join("") + "</ul>";
  }

  /* ---------- orchestration ---------- */

  function render() {
    var activities = filtered();
    renderSummary(activities);
    renderCategoryCards(activities);
    renderTable(activities);
    drawCharts();
    renderPeaks();
  }

  function init(data, peaks) {
    state.data = data;
    state.peaks = peaks;

    Object.keys(data.categories).forEach(function (category) {
      data.categories[category].forEach(function (sport) {
        state.sportCategory[sport] = category;
      });
    });

    state.allYears = Object.keys(data.activities.reduce(function (acc, a) {
      acc[a.date.slice(0, 4)] = true;
      return acc;
    }, {})).sort();

    peakPanel = $("hk-peak-detail");

    buildMonthOptions();
    buildCategoryToggles();
    wireControls();
    wireTableSorting();

    $("hk-updated").textContent = "Updated " + data.updated;
    $("hk-loading").style.display = "none";
    $("hk-content").style.display = "";

    applyPreset("all");
  }

  function start() {
    Promise.all([
      fetch(ACTIVITIES_URL).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      }),
      // The peaks file is optional: the rest of the page works without it.
      fetch(PEAKS_URL).then(function (r) {
        return r.ok ? r.json() : null;
      }).catch(function () { return null; })
    ]).then(function (results) {
      init(results[0], results[1]);
    }).catch(function (err) {
      $("hk-loading").textContent = "Could not load activity data: " + err.message;
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
    if (!state.peaks || !state.openPeak) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderPeaks, 150);
  });

  // Redraw on theme flip so chart text doesn't stay the old colour.
  var observer = new MutationObserver(function () {
    if (state.data) drawCharts();
  });
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"]
  });
})();
