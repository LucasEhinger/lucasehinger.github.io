// Per-person page (/bluewater/sailor/?id=…): shows one sailor's aggregate
// stats and full chronological sail history over a selectable month range.
// Relies on window.BW from bluewater-common.js.

(function () {
  "use strict";

  var data = null;
  var sailorIndex = -1;
  var sailorName = "";
  var fromYM = null;
  var toYM = null;

  // Sail-history table filters (role / type / status), plus the current
  // date-scoped rows they filter.
  var allRows = [];
  var fRole = "all"; // "all" | "s" | "c"
  var fType = "all"; // "all" | "race" | "pleasure"
  var fStatus = "all"; // "all" | "sailed" | "notselected" | "cancelled"
  var sortKey = "date"; // sort the history by "date" or "hours"
  var sortDir = -1; // -1 desc (newest / longest first), 1 asc

  // Tiles to show for the individual (subset/order of BW.METRICS).
  var TILE_KEYS = [
    "sails",
    "sail_time_hrs",
    "races",
    "pleasure",
    "as_skipper",
    "registrations",
  ];
  var TILE_LABEL = {};
  BW.METRICS.forEach(function (m) {
    TILE_LABEL[m.key] = m.label;
  });

  function qs(name) {
    var m = new RegExp("[?&]" + name + "=([^&]*)").exec(location.search);
    return m ? decodeURIComponent(m[1].replace(/\+/g, " ")) : null;
  }

  function validMonth(m) {
    return m && m >= data.date_min && m <= data.date_max;
  }

  function clampFrom(m) {
    return m < data.date_min ? data.date_min : m;
  }

  function presetRange(preset) {
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
      buttons[i].classList.toggle("active", r[0] === fromYM && r[1] === toYM);
    }
  }

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

  function roleLabel(role) {
    return role === "s" ? "Skipper" : "Crew";
  }

  // Canonical status: "cancelled" only when the sailor was selected and then
  // the event was cancelled ("x"); pending ("p") and unknown ("u") both mean
  // the sailor was not selected.
  function statusKey(st) {
    if (st === "c") return "sailed";
    if (st === "x") return "cancelled";
    return "notselected";
  }

  function statusBadge(st) {
    var key = statusKey(st);
    if (key === "sailed") return '<span class="bw-badge sail">Sailed</span>';
    if (key === "cancelled")
      return '<span class="bw-badge cancelled">Cancelled</span>';
    return '<span class="bw-badge pending">Not selected</span>';
  }

  function renderStats(stats) {
    var html = "";
    for (var i = 0; i < TILE_KEYS.length; i++) {
      var k = TILE_KEYS[i];
      html +=
        '<div class="bw-stat-tile"><div class="bw-stat-value">' +
        BW.fmt(stats[k], true) +
        '</div><div class="bw-stat-label">' +
        TILE_LABEL[k] +
        "</div></div>";
    }
    document.getElementById("bw-stat-row").innerHTML = html;
  }

  function renderRows(rows) {
    var parts = [];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var sailed = r.status === "c";
      var type = r.race ? "Race" : "Pleasure";
      var tripCell = r.eventId
        ? '<a href="http://sailing.mit.edu/calendar/events/event.php?id=' +
          encodeURIComponent(r.eventId) +
          '" target="_blank" rel="noopener" title="Open this event on the MIT sailing site">' +
          BW.escapeHtml(r.trip) +
          "</a>"
        : BW.escapeHtml(r.trip);
      parts.push(
        "<tr>" +
          "<td>" + r.date + "</td>" +
          '<td class="bw-trip">' + tripCell + "</td>" +
          "<td>" + roleLabel(r.role) + "</td>" +
          '<td class="bw-num">' + (sailed ? BW.fmt(r.hours, true) : "—") + "</td>" +
          "<td>" + type + "</td>" +
          "<td>" + statusBadge(r.status) + "</td>" +
          "</tr>"
      );
    }
    if (!parts.length) {
      parts.push(
        '<tr><td colspan="6" style="font-style:italic;opacity:0.7;">No sail records match the current filters.</td></tr>'
      );
    }
    document.getElementById("bw-tbody").innerHTML = parts.join("");
  }

  function updateSortArrows() {
    var ths = document.querySelectorAll("th.bw-sortable");
    for (var i = 0; i < ths.length; i++) {
      var arrow = ths[i].querySelector(".bw-sort-arrow");
      if (!arrow) continue;
      arrow.textContent =
        ths[i].getAttribute("data-key") === sortKey
          ? sortDir === -1
            ? " ▼"
            : " ▲"
          : "";
    }
  }

  // Apply the role/type/status filters + current sort to the date-scoped rows.
  function applyFilters() {
    var rows = allRows.filter(function (r) {
      if (fRole !== "all" && r.role !== fRole) return false;
      if (fType !== "all" && (r.race ? "race" : "pleasure") !== fType)
        return false;
      if (fStatus !== "all" && statusKey(r.status) !== fStatus) return false;
      return true;
    });
    rows.sort(function (a, b) {
      if (sortKey === "hours") return (a.hours - b.hours) * sortDir;
      return (a.date < b.date ? -1 : a.date > b.date ? 1 : 0) * sortDir;
    });
    renderRows(rows);
    var heading = document.getElementById("bw-history-heading");
    if (heading) {
      var filtered = rows.length !== allRows.length;
      heading.textContent =
        "Sail history — " +
        (filtered
          ? rows.length + " of " + allRows.length + " events"
          : allRows.length + " events");
    }
  }

  function renderLastUpdate() {
    var el = document.getElementById("bw-last-update");
    el.textContent =
      "Showing " + BW.monthLabel(fromYM) + " – " + BW.monthLabel(toYM);
  }

  // Keep the URL in sync so a filtered view is shareable / bookmarkable.
  function updateUrl() {
    var id = data.sailors[sailorIndex].id;
    var q =
      "?id=" +
      encodeURIComponent(id) +
      "&from=" +
      fromYM +
      "&to=" +
      toYM;
    history.replaceState(null, "", location.pathname + q);
  }

  function recompute() {
    if (fromYM > toYM) {
      var t = fromYM;
      fromYM = toYM;
      toYM = t;
      document.getElementById("bw-from").value = fromYM;
      document.getElementById("bw-to").value = toYM;
    }
    var res = BW.person(data, sailorIndex, fromYM, toYM);
    allRows = res.rows;
    markActivePreset();
    renderStats(res.stats);
    applyFilters();
    renderLastUpdate();
    updateUrl();
  }

  function showNotFound() {
    document.getElementById("bw-status").textContent =
      "Sailor not found. Please return to the leaderboard and pick a name.";
    document.getElementById("bw-sailor-name").textContent = "Unknown sailor";
  }

  function init(loaded) {
    data = loaded;
    var id = qs("id");
    sailorIndex = id ? BW.findSailorIndexById(data, id) : -1;
    if (sailorIndex < 0) {
      showNotFound();
      return;
    }
    sailorName = data.sailors[sailorIndex].n;
    document.getElementById("bw-sailor-name").textContent = sailorName;
    document.title = sailorName + " — MIT Bluewater";

    document.getElementById("bw-status").style.display = "none";
    document.getElementById("bw-content").style.display = "";

    var urlFrom = qs("from");
    var urlTo = qs("to");
    fromYM = validMonth(urlFrom) ? urlFrom : data.date_min;
    toYM = validMonth(urlTo) ? urlTo : data.date_max;

    fillMonthSelect(document.getElementById("bw-from"), fromYM);
    fillMonthSelect(document.getElementById("bw-to"), toYM);
    recompute();

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

    // Role / type / status filters just re-render the (already date-scoped) table.
    document.getElementById("bw-f-role").addEventListener("change", function (e) {
      fRole = e.target.value;
      applyFilters();
    });
    document.getElementById("bw-f-type").addEventListener("change", function (e) {
      fType = e.target.value;
      applyFilters();
    });
    document
      .getElementById("bw-f-status")
      .addEventListener("change", function (e) {
        fStatus = e.target.value;
        applyFilters();
      });

    // Sortable history headers (Date, Hours).
    var sortables = document.querySelectorAll("th.bw-sortable");
    for (var si = 0; si < sortables.length; si++) {
      sortables[si].addEventListener("click", function (e) {
        var key = e.currentTarget.getAttribute("data-key");
        if (sortKey === key) sortDir = -sortDir;
        else {
          sortKey = key;
          sortDir = -1; // new column defaults to descending
        }
        updateSortArrows();
        applyFilters();
      });
    }
    updateSortArrows();
  }

  document.addEventListener("DOMContentLoaded", function () {
    BW.load(init, function (err) {
      console.error("Failed to load Bluewater data:", err);
      document.getElementById("bw-status").textContent =
        "Sorry — the sailing data failed to load. Please try again later.";
    });
  });
})();
