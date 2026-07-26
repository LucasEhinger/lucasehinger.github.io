// Leaderboard page (/bluewater/): date-range control, client-side aggregation,
// sortable/searchable table with links to per-person pages, and a Top-15 chart.
// Relies on window.BW from bluewater-common.js.

(function () {
  "use strict";

  var TOP_N = 15;

  var data = null;
  var fromYM = null;
  var toYM = null;
  var board = []; // aggregated leaderboard for the current range
  var sortKey = "sails"; // default sort: number of sails
  var sortDir = -1; // -1 desc, 1 asc
  var searchTerm = "";
  var crewFilter = null; // {sailorId: true} when an upcoming sail is selected
  var upcomingEvents = []; // future-dated events, soonest first
  var excludeConfirmed = false; // hide skipper + confirmed crew from the filter
  var sailMarks = null; // {sailorId: "confirmed" | "skipper"} for the selected sail
  var chartSailors = []; // sailors in chart-bar order, for click-to-navigate
  var chartClickBound = false; // whether the plotly_click handler is attached

  var yearly = null; // program-wide per-year aggregation (all history)
  var yearFrom = null; // year range shown as lines in the monthly plots
  var yearTo = null;
  var DEFAULT_YEAR_SPAN = 5; // default number of recent years shown

  var COLOR_PLEASURE = "#1f77b4";
  var COLOR_RACES = "#ff7f0e";
  var COLOR_SAILORS = "#2ca02c";

  var COLUMNS = [{ key: "name", label: "Sailor", numeric: false }].concat(
    BW.METRICS.map(function (m) {
      return { key: m.key, label: m.label, numeric: true };
    })
  );

  // ---- range helpers -------------------------------------------------------

  function clampFrom(m) {
    return m < data.date_min ? data.date_min : m;
  }

  function presetRange(preset) {
    if (preset === "all") return [data.date_min, data.date_max];
    // "Last N" windows end at the current month, not date_max (which runs a
    // couple months into the future to capture upcoming sails).
    var end = BW.currentMonth();
    if (end > data.date_max) end = data.date_max;
    if (preset === "12") return [clampFrom(BW.addMonths(end, -11)), end];
    if (preset === "60") return [clampFrom(BW.addMonths(end, -59)), end];
    if (preset === "ytd") return [clampFrom(end.slice(0, 4) + "-01"), end];
    return [data.date_min, data.date_max];
  }

  function markActivePreset() {
    var buttons = document.querySelectorAll("#bw-presets button");
    for (var i = 0; i < buttons.length; i++) {
      var r = presetRange(buttons[i].getAttribute("data-preset"));
      buttons[i].classList.toggle(
        "active",
        r[0] === fromYM && r[1] === toYM
      );
    }
  }

  // ---- population ----------------------------------------------------------

  function fillMonthSelect(select, selected) {
    var html = "";
    var m = data.date_min;
    while (m <= data.date_max) {
      html +=
        '<option value="' +
        m +
        '"' +
        (m === selected ? " selected" : "") +
        ">" +
        BW.monthLabel(m) +
        "</option>";
      m = BW.addMonths(m, 1);
    }
    select.innerHTML = html;
  }

  function fillMetricSelect() {
    var sel = document.getElementById("bw-metric");
    var html = "";
    for (var i = 0; i < BW.METRICS.length; i++) {
      html +=
        '<option value="' +
        BW.METRICS[i].key +
        '">' +
        BW.METRICS[i].label +
        "</option>";
    }
    html += '<option value="sails_registrations">Sails + Registrations</option>';
    sel.innerHTML = html;
    sel.value = "sails";
  }

  // ---- rendering -----------------------------------------------------------

  function filtered() {
    var rows = board;
    if (crewFilter) {
      rows = rows.filter(function (s) {
        return crewFilter[s.id];
      });
    }
    if (searchTerm) {
      var term = searchTerm.toLowerCase();
      rows = rows.filter(function (s) {
        return s.name.toLowerCase().indexOf(term) !== -1;
      });
    }
    return rows;
  }

  function sorted(rows) {
    var copy = rows.slice();
    copy.sort(function (a, b) {
      var av = a[sortKey];
      var bv = b[sortKey];
      if (typeof av === "string") return av.localeCompare(bv) * sortDir;
      return (av - bv) * sortDir;
    });
    return copy;
  }

  function renderStats() {
    var totalHours = 0,
      totalSails = 0,
      totalRaces = 0;
    for (var i = 0; i < board.length; i++) {
      totalHours += board[i].sail_time_hrs;
      totalSails += board[i].sails;
      totalRaces += board[i].races;
    }
    var tiles = [
      { value: board.length.toLocaleString("en-US"), label: "Sailors" },
      {
        value: Math.round(totalHours).toLocaleString("en-US"),
        label: "Total sail-hours",
      },
      { value: totalSails.toLocaleString("en-US"), label: "Total sails" },
      { value: totalRaces.toLocaleString("en-US"), label: "Total races" },
    ];
    var html = "";
    for (var j = 0; j < tiles.length; j++) {
      html +=
        '<div class="bw-stat-tile"><div class="bw-stat-value">' +
        tiles[j].value +
        '</div><div class="bw-stat-label">' +
        tiles[j].label +
        "</div></div>";
    }
    document.getElementById("bw-stat-row").innerHTML = html;
  }

  function renderHead() {
    var row = document.getElementById("bw-thead-row");
    var html = '<th data-key="rank">#</th>';
    for (var i = 0; i < COLUMNS.length; i++) {
      var col = COLUMNS[i];
      var arrow =
        col.key === sortKey
          ? ' <span class="bw-sort-arrow">' +
            (sortDir === -1 ? "▼" : "▲") +
            "</span>"
          : "";
      html +=
        '<th class="bw-sortable" data-key="' +
        col.key +
        '">' +
        BW.escapeHtml(col.label) +
        arrow +
        "</th>";
    }
    row.innerHTML = html;
    var headers = row.querySelectorAll("th.bw-sortable");
    for (var h = 0; h < headers.length; h++) {
      headers[h].addEventListener("click", onHeaderClick);
    }
  }

  function onHeaderClick(e) {
    var key = e.currentTarget.getAttribute("data-key");
    var col = COLUMNS.filter(function (c) {
      return c.key === key;
    })[0];
    if (!col) return;
    if (sortKey === key) sortDir = -sortDir;
    else {
      sortKey = key;
      sortDir = col.numeric ? -1 : 1;
    }
    renderHead();
    renderTable();
  }

  function sailorUrl(s) {
    return (
      "/bluewater/sailor/?id=" +
      encodeURIComponent(s.id) +
      "&from=" +
      fromYM +
      "&to=" +
      toYM
    );
  }

  function sailorLink(s) {
    return '<a href="' + sailorUrl(s) + '">' + BW.escapeHtml(s.name) + "</a>";
  }

  function renderTable() {
    var rows = sorted(filtered());
    var parts = [];
    for (var i = 0; i < rows.length; i++) {
      var s = rows[i];
      var cells = '<td class="bw-rank">' + (i + 1) + "</td>";
      for (var c = 0; c < COLUMNS.length; c++) {
        var col = COLUMNS[c];
        if (col.key === "name") {
          var mark = "";
          if (sailMarks && sailMarks[s.id]) {
            mark =
              sailMarks[s.id] === "skipper"
                ? '<span class="bw-mark" title="Skipper">⛵</span> '
                : '<span class="bw-mark" title="Confirmed">✓</span> ';
          }
          cells += '<td class="bw-name">' + mark + sailorLink(s) + "</td>";
        } else {
          cells += "<td>" + BW.fmt(s[col.key], col.numeric) + "</td>";
        }
      }
      parts.push("<tr>" + cells + "</tr>");
    }
    document.getElementById("bw-tbody").innerHTML = parts.join("");
  }

  // When filtering to a specific sail's crew, show everyone (grow the chart to
  // fit); otherwise fall back to the fixed-height Top-15 view.
  // Returns the pixel height so callers can pin layout.height to match. Plotly's
  // responsive autosize does NOT re-read a programmatic container-height change on
  // Plotly.react, so without an explicit layout.height the plot keeps its old
  // height — the bars then cram together, out of line with the y-axis labels.
  function setChartHeight(count, perBar, showingAll) {
    var h = showingAll ? Math.max(460, count * perBar + 90) : 460;
    document.getElementById("bw-chart").style.height = h + "px";
    return h;
  }

  // Chart y-axis label with a check-mark when the sailor is confirmed/skipper
  // for the selected upcoming sail. Plotly renders a small subset of HTML in
  // tick labels, so we can color the check green.
  function chartLabel(s) {
    if (!sailMarks || !sailMarks[s.id]) return s.name;
    var check = '<span style="color:#2ca02c">✓</span>';
    return (sailMarks[s.id] === "skipper" ? "⛵" : check) + " " + s.name;
  }

  // Make chart bars navigate to the sailor's page, like the table links.
  // Bound once; Plotly.react re-renders reuse the same graph div, and
  // chartSailors is refreshed on every render to stay in sync with the bars.
  function bindChartClick() {
    if (chartClickBound) return;
    var gd = document.getElementById("bw-chart");
    if (!gd || typeof gd.on !== "function") return;
    gd.on("plotly_click", function (ev) {
      if (!ev || !ev.points || !ev.points.length) return;
      var s = chartSailors[ev.points[0].pointNumber];
      if (s) window.location.href = sailorUrl(s);
    });
    gd.on("plotly_hover", function () {
      gd.style.cursor = "pointer";
    });
    gd.on("plotly_unhover", function () {
      gd.style.cursor = "";
    });
    chartClickBound = true;
  }

  function renderChart() {
    var metric = document.getElementById("bw-metric").value;
    if (metric === "sails_registrations") {
      renderComboChart();
      return;
    }
    var showingAll = !!crewFilter;
    var rows = filtered()
      .slice()
      .sort(function (a, b) {
        return b[metric] - a[metric];
      });
    if (!showingAll) rows = rows.slice(0, TOP_N);
    var chartH = setChartHeight(rows.length, 26, showingAll);
    chartSailors = rows.slice().reverse();
    var names = rows.map(chartLabel).reverse();
    var values = rows
      .map(function (s) {
        return s[metric];
      })
      .reverse();

    var textColor = BW.chartTextColor();
    var label = (BW.METRICS.filter(function (m) {
      return m.key === metric;
    })[0] || {}).label;

    var trace = {
      type: "bar",
      orientation: "h",
      x: values,
      y: names,
      marker: { color: "#1f77b4" },
      hovertemplate: "%{y}: %{x}<extra></extra>",
    };
    var layout = {
      title: {
        text: (showingAll ? "Registered sailors" : "Top sailors") + " (" + label + ")",
        font: { color: textColor },
      },
      margin: { l: 110, r: 20, t: 44, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        title: { text: label, font: { color: textColor } },
        tickfont: { color: textColor },
        gridcolor: "rgba(128,128,128,0.2)",
        zeroline: false,
      },
      yaxis: { tickfont: { color: textColor }, automargin: true },
    };
    layout.height = chartH;
    Plotly.react("bw-chart", [trace], layout, {
      displayModeBar: false,
      responsive: true,
    });
    bindChartClick();
  }

  // Grouped bars showing each top sailor's sails and registrations side by side.
  function renderComboChart() {
    var showingAll = !!crewFilter;
    var rows = filtered()
      .slice()
      .sort(function (a, b) {
        return b.sails - a.sails;
      });
    if (!showingAll) rows = rows.slice(0, TOP_N);
    var chartH = setChartHeight(rows.length, 42, showingAll);
    chartSailors = rows.slice().reverse();
    var names = rows.map(chartLabel).reverse();
    var sails = rows
      .map(function (s) {
        return s.sails;
      })
      .reverse();
    var regs = rows
      .map(function (s) {
        return s.registrations;
      })
      .reverse();

    var textColor = BW.chartTextColor();
    var traces = [
      {
        type: "bar",
        orientation: "h",
        name: "Sails",
        y: names,
        x: sails,
        marker: { color: COLOR_PLEASURE },
        hovertemplate: "%{y} — Sails: %{x}<extra></extra>",
      },
      {
        type: "bar",
        orientation: "h",
        name: "Registrations",
        y: names,
        x: regs,
        marker: { color: COLOR_RACES },
        hovertemplate: "%{y} — Registrations: %{x}<extra></extra>",
      },
    ];
    var layout = {
      title: {
        text:
          (showingAll ? "Registered sailors" : "Top sailors") +
          " (Sails vs Registrations)",
        font: { color: textColor },
      },
      barmode: "group",
      bargap: 0.25,
      margin: { l: 110, r: 20, t: 44, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        title: { text: "Count", font: { color: textColor } },
        tickfont: { color: textColor },
        gridcolor: "rgba(128,128,128,0.2)",
        zeroline: false,
      },
      yaxis: { tickfont: { color: textColor }, automargin: true },
      legend: { font: { color: textColor }, orientation: "h" },
    };
    layout.height = chartH;
    Plotly.react("bw-chart", traces, layout, {
      displayModeBar: false,
      responsive: true,
    });
    bindChartClick();
  }

  function renderLastUpdate() {
    document.getElementById("bw-last-update").textContent =
      "Showing " + BW.monthLabel(fromYM) + " – " + BW.monthLabel(toYM);
  }

  // "Last updated" text shown next to the Refresh button.
  function renderUpdated() {
    var el = document.getElementById("bw-updated");
    if (!el || !data.last_updated) return;
    var d = new Date(data.last_updated);
    if (isNaN(d.getTime())) return;
    el.textContent =
      "Updated " +
      d.toLocaleString("en-US", {
        timeZone: "America/New_York",
        year: "numeric",
        month: "short",
        day: "2-digit",
      }) +
      " ET";
  }

  // ---- upcoming-sail crew filter -------------------------------------------

  function todayISO() {
    var d = new Date();
    var m = d.getMonth() + 1;
    var day = d.getDate();
    return (
      d.getFullYear() +
      "-" +
      (m < 10 ? "0" + m : m) +
      "-" +
      (day < 10 ? "0" + day : day)
    );
  }

  // Drop the "MIT Sailing:" prefix from event titles for display.
  function cleanTitle(t) {
    return (t || "").replace(/^\s*MIT Sailing:\s*/i, "").trim();
  }

  // Populate the "Upcoming sail" dropdown with future-dated events, soonest first.
  function fillSailSelect() {
    var today = todayISO();
    upcomingEvents = data.events
      .filter(function (ev) {
        return ev.d >= today;
      })
      .sort(function (a, b) {
        return a.d < b.d ? -1 : a.d > b.d ? 1 : 0;
      });

    var sel = document.getElementById("bw-sail");
    var details = document.getElementById("bw-sail-details");
    if (!upcomingEvents.length) {
      // Nothing to filter by — hide the whole disclosure.
      if (details) details.style.display = "none";
      sel.innerHTML = '<option value="">No upcoming sails scheduled</option>';
      sel.disabled = true;
      return;
    }
    if (details) details.style.display = "";
    sel.disabled = false;
    var html = '<option value="">— All sailors —</option>';
    for (var i = 0; i < upcomingEvents.length; i++) {
      var ev = upcomingEvents[i];
      html +=
        '<option value="' + i + '">' +
        BW.escapeHtml(ev.d + " — " + cleanTitle(ev.t) + " (" + ev.p.length + ")") +
        "</option>";
    }
    sel.innerHTML = html;
  }

  // Apply (or clear) the crew filter for a selected upcoming sail.
  function applySail(value) {
    var hasSail = !(value === "" || value == null);
    var excludeBtn = document.getElementById("bw-sail-exclude");
    if (!hasSail) {
      // No sail selected: clear the filter and hide/reset the exclude toggle.
      crewFilter = null;
      sailMarks = null;
      excludeConfirmed = false;
      if (excludeBtn) {
        excludeBtn.style.display = "none";
        excludeBtn.classList.remove("active");
      }
    } else {
      if (excludeBtn) excludeBtn.style.display = "";
      var ev = upcomingEvents[parseInt(value, 10)];
      crewFilter = {};
      sailMarks = {};
      for (var i = 0; i < ev.p.length; i++) {
        var id = data.sailors[ev.p[i][0]].id;
        var role = ev.p[i][1];
        var st = ev.p[i][2];
        // Confirmed crew and the skipper get a check-mark.
        if (st === "c") sailMarks[id] = role === "s" ? "skipper" : "confirmed";
        // Optionally hide skipper + confirmed crew (status "c").
        if (excludeConfirmed && st === "c") continue;
        crewFilter[id] = true;
      }
    }
    renderTable();
    renderChart();
    renderSailNote(value);
  }

  function renderSailNote(value) {
    var note = document.getElementById("bw-sail-note");
    if (value === "" || value == null) {
      note.style.display = "none";
      note.innerHTML = "";
      return;
    }
    var ev = upcomingEvents[parseInt(value, 10)];
    var count = filtered().length;
    var who = excludeConfirmed ? " not-yet-confirmed sailor" : " sailor";
    note.style.display = "";
    note.innerHTML =
      "Showing <strong>" +
      count +
      "</strong>" +
      who +
      (count === 1 ? "" : "s") +
      " registered for " +
      BW.escapeHtml(ev.d + " — " + cleanTitle(ev.t)) +
      '<span class="bw-clear-sail" role="button" tabindex="0">clear</span>';
    var clear = note.querySelector(".bw-clear-sail");
    clear.addEventListener("click", function () {
      document.getElementById("bw-sail").value = "";
      applySail("");
    });
  }

  // ---- program-wide trend charts -------------------------------------------

  // Sailor indices that actually sailed (status "c") on an event.
  function sailedIdx(ev) {
    var out = [];
    for (var i = 0; i < ev.p.length; i++) {
      if (ev.p[i][2] === "c") out.push(ev.p[i][0]);
    }
    return out;
  }

  // Per-year: count of sail outings (race vs pleasure) and unique sailors.
  function computeYearly() {
    var by = {};
    for (var i = 0; i < data.events.length; i++) {
      var ev = data.events[i];
      var idx = sailedIdx(ev);
      if (!idx.length) continue; // outing didn't take place
      var y = ev.d.slice(0, 4);
      var b = by[y] || (by[y] = { races: 0, pleasure: 0, sailors: {} });
      if (ev.r) b.races++;
      else b.pleasure++;
      for (var k = 0; k < idx.length; k++) b.sailors[idx[k]] = 1;
    }
    var years = Object.keys(by).sort();
    return {
      years: years,
      races: years.map(function (y) { return by[y].races; }),
      pleasure: years.map(function (y) { return by[y].pleasure; }),
      sailors: years.map(function (y) {
        return Object.keys(by[y].sailors).length;
      }),
    };
  }

  // Per-month for a given year (12 buckets, Jan..Dec).
  function computeMonthly(year) {
    var buckets = [];
    for (var m = 0; m < 12; m++) buckets.push({ races: 0, pleasure: 0, sailors: {} });
    for (var i = 0; i < data.events.length; i++) {
      var ev = data.events[i];
      if (ev.d.slice(0, 4) !== year) continue;
      var idx = sailedIdx(ev);
      if (!idx.length) continue;
      var b = buckets[parseInt(ev.d.slice(5, 7), 10) - 1];
      if (ev.r) b.races++;
      else b.pleasure++;
      for (var k = 0; k < idx.length; k++) b.sailors[idx[k]] = 1;
    }
    return {
      races: buckets.map(function (b) { return b.races; }),
      pleasure: buckets.map(function (b) { return b.pleasure; }),
      sailors: buckets.map(function (b) { return Object.keys(b.sailors).length; }),
    };
  }

  function yearsInRange() {
    var out = [];
    for (var i = 0; i < yearly.years.length; i++) {
      var y = yearly.years[i];
      if (y >= yearFrom && y <= yearTo) out.push(y);
    }
    return out;
  }

  function lineLayout(title, textColor) {
    return {
      title: { text: title, font: { color: textColor } },
      margin: { l: 48, r: 16, t: 44, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        type: "category",
        tickfont: { color: textColor },
        gridcolor: "rgba(128,128,128,0.15)",
        zeroline: false,
      },
      yaxis: {
        tickfont: { color: textColor },
        gridcolor: "rgba(128,128,128,0.2)",
        zeroline: false,
        rangemode: "tozero",
      },
      legend: { font: { color: textColor } },
      showlegend: true,
      hovermode: "x unified",
      hoverlabel: BW.chartHoverLabel(textColor),
    };
  }

  function trendLayout(title, textColor, stacked) {
    return {
      title: { text: title, font: { color: textColor } },
      barmode: stacked ? "stack" : "group",
      margin: { l: 48, r: 16, t: 44, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        type: "category",
        tickfont: { color: textColor },
        gridcolor: "rgba(128,128,128,0.15)",
        zeroline: false,
        // Label every year (Plotly otherwise thins them to ~every 2-3 years,
        // making it hard to tell which bar is which). Rotate so they all fit,
        // and draw outward tick marks that connect each label to its bar.
        tickmode: "linear",
        tick0: 0,
        dtick: 1,
        tickangle: -45,
        ticks: "outside",
        ticklen: 5,
        tickcolor: "rgba(128,128,128,0.5)",
        automargin: true,
      },
      yaxis: {
        tickfont: { color: textColor },
        gridcolor: "rgba(128,128,128,0.2)",
        zeroline: false,
        rangemode: "tozero",
      },
      // Anchor the legend below the (now taller, rotated) year labels so the
      // Races/Pleasure markers don't overlap them.
      legend: {
        font: { color: textColor },
        orientation: "h",
        x: 0.5,
        xanchor: "center",
        y: -0.25,
        yanchor: "top",
      },
      showlegend: stacked,
      hoverlabel: BW.chartHoverLabel(textColor),
    };
  }

  function renderProgramCharts() {
    if (!yearly) return;
    var tc = BW.chartTextColor();
    var cfg = { displayModeBar: false, responsive: true };

    Plotly.react(
      "bw-plot-sails-year",
      [
        { type: "bar", name: "Pleasure", x: yearly.years, y: yearly.pleasure, marker: { color: COLOR_PLEASURE } },
        { type: "bar", name: "Races", x: yearly.years, y: yearly.races, marker: { color: COLOR_RACES } },
      ],
      trendLayout("Sails per year", tc, true),
      cfg
    );

    Plotly.react(
      "bw-plot-sailors-year",
      [{ type: "bar", name: "Sailors", x: yearly.years, y: yearly.sailors, marker: { color: COLOR_SAILORS } }],
      trendLayout("Unique sailors per year", tc, false),
      cfg
    );

    // Monthly plots: one line per year across the selected year range so the
    // seasonal shape can be compared year to year.
    var years = yearsInRange();
    var sailsTraces = [];
    var sailorTraces = [];
    for (var yi = 0; yi < years.length; yi++) {
      var mo = computeMonthly(years[yi]);
      var sailsPerMonth = mo.races.map(function (r, i) {
        return r + mo.pleasure[i];
      });
      sailsTraces.push({
        type: "scatter",
        mode: "lines+markers",
        name: years[yi],
        x: BW.MONTHS,
        y: sailsPerMonth,
      });
      sailorTraces.push({
        type: "scatter",
        mode: "lines+markers",
        name: years[yi],
        x: BW.MONTHS,
        y: mo.sailors,
      });
    }
    Plotly.react("bw-plot-sails-month", sailsTraces, lineLayout("Sails per month by year", tc), cfg);
    Plotly.react("bw-plot-sailors-month", sailorTraces, lineLayout("Unique sailors per month by year", tc), cfg);
  }

  // Build the program section: aggregate, fill the year-range selects, render.
  function initProgram(preserve) {
    yearly = computeYearly();
    document.getElementById("bw-trends").style.display = "";

    var years = yearly.years;
    var opts = "";
    for (var i = 0; i < years.length; i++) {
      opts += '<option value="' + years[i] + '">' + years[i] + "</option>";
    }
    var fromSel = document.getElementById("bw-year-from");
    var toSel = document.getElementById("bw-year-to");
    fromSel.innerHTML = opts;
    toSel.innerHTML = opts;

    var latest = years[years.length - 1];
    if (!(preserve && years.indexOf(yearTo) !== -1)) yearTo = latest;
    if (!(preserve && years.indexOf(yearFrom) !== -1)) {
      yearFrom = years[Math.max(0, years.length - DEFAULT_YEAR_SPAN)];
    }
    if (yearFrom > yearTo) yearFrom = yearTo;
    fromSel.value = yearFrom;
    toSel.value = yearTo;
    renderProgramCharts();
  }

  function reTheme() {
    renderChart();
    renderProgramCharts();
  }

  // ---- recompute + wiring --------------------------------------------------

  function recompute() {
    if (fromYM > toYM) {
      var t = fromYM;
      fromYM = toYM;
      toYM = t;
      document.getElementById("bw-from").value = fromYM;
      document.getElementById("bw-to").value = toYM;
    }
    board = BW.leaderboard(data, fromYM, toYM);
    markActivePreset();
    renderStats();
    renderTable();
    renderChart();
    renderLastUpdate();
  }

  function init(loaded) {
    data = loaded;
    document.getElementById("bw-status").style.display = "none";
    var wrap = document.querySelector(".bw-table-wrap");
    if (wrap) wrap.style.display = "";

    // Default the leaderboard to the last 12 months rather than all-time.
    var initialRange = presetRange("12");
    fromYM = initialRange[0];
    toYM = initialRange[1];

    fillMetricSelect();
    fillMonthSelect(document.getElementById("bw-from"), fromYM);
    fillMonthSelect(document.getElementById("bw-to"), toYM);
    renderHead();
    recompute();
    renderUpdated();
    initProgram(false);
    fillSailSelect();

    document.getElementById("bw-sail").addEventListener("change", function (e) {
      applySail(e.target.value);
    });

    var excludeBtn = document.getElementById("bw-sail-exclude");
    excludeBtn.addEventListener("click", function () {
      excludeConfirmed = !excludeConfirmed;
      excludeBtn.classList.toggle("active", excludeConfirmed);
      applySail(document.getElementById("bw-sail").value);
    });

    document.getElementById("bw-year-from").addEventListener("change", function (e) {
      yearFrom = e.target.value;
      if (yearFrom > yearTo) {
        yearTo = yearFrom;
        document.getElementById("bw-year-to").value = yearTo;
      }
      renderProgramCharts();
    });
    document.getElementById("bw-year-to").addEventListener("change", function (e) {
      yearTo = e.target.value;
      if (yearTo < yearFrom) {
        yearFrom = yearTo;
        document.getElementById("bw-year-from").value = yearFrom;
      }
      renderProgramCharts();
    });

    // Presets
    var presetButtons = document.querySelectorAll("#bw-presets button");
    for (var i = 0; i < presetButtons.length; i++) {
      presetButtons[i].addEventListener("click", function (e) {
        var r = presetRange(e.currentTarget.getAttribute("data-preset"));
        fromYM = r[0];
        toYM = r[1];
        document.getElementById("bw-from").value = fromYM;
        document.getElementById("bw-to").value = toYM;
        recompute();
      });
    }

    document.getElementById("bw-from").addEventListener("change", function (e) {
      fromYM = e.target.value;
      recompute();
    });
    document.getElementById("bw-to").addEventListener("change", function (e) {
      toYM = e.target.value;
      recompute();
    });
    document
      .getElementById("bw-metric")
      .addEventListener("change", renderChart);

    var searchInput = document.getElementById("bw-search");
    var debounce;
    searchInput.addEventListener("input", function () {
      searchTerm = searchInput.value.trim();
      clearTimeout(debounce);
      debounce = setTimeout(function () {
        renderTable();
        renderChart();
      }, 120);
    });

    // Re-theme all charts when the OS scheme or the site's theme toggle changes.
    if (window.matchMedia) {
      window
        .matchMedia("(prefers-color-scheme: dark)")
        .addEventListener("change", reTheme);
    }
    new MutationObserver(reTheme).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    // Refresh button: re-fetch the latest published data (bypassing cache).
    var refreshBtn = document.getElementById("bw-refresh");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", function () {
        var orig = refreshBtn.textContent;
        refreshBtn.disabled = true;
        refreshBtn.textContent = "⟳ Refreshing…";
        BW.load(
          function (loaded) {
            data = loaded;
            if (fromYM < data.date_min || fromYM > data.date_max)
              fromYM = data.date_min;
            if (toYM < data.date_min || toYM > data.date_max)
              toYM = data.date_max;
            fillMonthSelect(document.getElementById("bw-from"), fromYM);
            fillMonthSelect(document.getElementById("bw-to"), toYM);
            recompute();
            renderUpdated();
            initProgram(true);
            fillSailSelect();
            applySail(""); // reset crew filter (upcoming events may have changed)
            refreshBtn.disabled = false;
            refreshBtn.textContent = orig;
          },
          function () {
            refreshBtn.disabled = false;
            refreshBtn.textContent = "⟳ Refresh failed";
          },
          true
        );
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    BW.load(init, function (err) {
      console.error("Failed to load Bluewater stats:", err);
      var status = document.getElementById("bw-status");
      if (status)
        status.textContent =
          "Sorry — the sailing stats failed to load. Please try again later.";
    });
  });
})();
