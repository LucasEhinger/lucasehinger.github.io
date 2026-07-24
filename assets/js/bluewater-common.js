// Shared helpers for the MIT Bluewater pages (/bluewater/ and
// /bluewater/sailor/). Loads the event-level data file and provides the
// client-side aggregation used to build the leaderboard and per-person
// histories over an arbitrary month range.
//
// Data schema (files/bluewater/bluewater_data.json):
//   sailors: [ {id, n}, ... ]            (array index === participant ref)
//   events:  [ {d,t,h,r,p:[[idx,role,st],...]}, ... ]
//     role: "s" skipper | "c" crew ;  st: "c" sail | "p" pending | "x" cancelled | "u" unknown

(function () {
  "use strict";

  var MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];

  // Metric columns (also drive the chart dropdown). Order = table column order.
  var METRICS = [
    { key: "sails", label: "Sails" },
    { key: "sail_time_hrs", label: "Sail time (hrs)" },
    { key: "races", label: "Races" },
    { key: "pleasure", label: "Pleasure" },
    { key: "registrations", label: "Registrations" },
    { key: "multi_day", label: "Multi-day" },
    { key: "full_day", label: "Full-day (6+ hr)" },
    { key: "as_skipper", label: "As skipper" },
  ];

  function emptyStats() {
    return {
      registrations: 0,
      sails: 0,
      races: 0,
      pleasure: 0,
      multi_day: 0,
      full_day: 0,
      as_skipper: 0,
      sail_time_hrs: 0,
    };
  }

  // Fold one (event, participant) record into a stats accumulator.
  function accumulate(stats, ev, role, st) {
    stats.registrations += 1;
    if (role === "s") stats.as_skipper += 1;
    if (st === "c") {
      stats.sails += 1;
      if (ev.r) stats.races += 1;
      else stats.pleasure += 1;
      if (ev.h > 24) stats.multi_day += 1;
      if (ev.h > 6) stats.full_day += 1;
      stats.sail_time_hrs += ev.h;
    }
  }

  function ym(ev) {
    return ev.d ? ev.d.slice(0, 7) : "";
  }

  function inRange(ev, fromYM, toYM) {
    var m = ym(ev);
    if (fromYM && m < fromYM) return false;
    if (toYM && m > toYM) return false;
    return true;
  }

  var BW = {
    DATA_URL: "/files/bluewater/bluewater_data.json",
    MONTHS: MONTHS,
    METRICS: METRICS,

    load: function (onOk, onErr, bust) {
      var url = bust ? BW.DATA_URL + "?t=" + Date.now() : BW.DATA_URL;
      fetch(url, bust ? { cache: "no-store" } : undefined)
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(onOk)
        .catch(onErr);
    },

    // "2023-06" -> "Jun 2023"
    monthLabel: function (m) {
      if (!m) return "";
      var parts = m.split("-");
      return MONTHS[parseInt(parts[1], 10) - 1] + " " + parts[0];
    },

    // Current calendar month as "YYYY-MM" (local time).
    currentMonth: function () {
      var d = new Date();
      var m = d.getMonth() + 1;
      return d.getFullYear() + "-" + (m < 10 ? "0" + m : "" + m);
    },

    // Add n months to a "YYYY-MM" string.
    addMonths: function (m, n) {
      var parts = m.split("-");
      var idx = parseInt(parts[0], 10) * 12 + (parseInt(parts[1], 10) - 1) + n;
      var y = Math.floor(idx / 12);
      var mo = (idx % 12) + 1;
      return y + "-" + (mo < 10 ? "0" + mo : "" + mo);
    },

    inRange: inRange,

    // Leaderboard: one stats row per sailor active in the range.
    leaderboard: function (data, fromYM, toYM) {
      var acc = {}; // index -> stats
      var events = data.events;
      for (var i = 0; i < events.length; i++) {
        var ev = events[i];
        if (!inRange(ev, fromYM, toYM)) continue;
        var p = ev.p;
        for (var j = 0; j < p.length; j++) {
          var idx = p[j][0];
          var s = acc[idx] || (acc[idx] = emptyStats());
          accumulate(s, ev, p[j][1], p[j][2]);
        }
      }
      var out = [];
      for (var key in acc) {
        if (!acc.hasOwnProperty(key)) continue;
        var stats = acc[key];
        stats.sail_time_hrs = Math.round(stats.sail_time_hrs * 100) / 100;
        stats.id = data.sailors[key].id;
        stats.name = data.sailors[key].n;
        out.push(stats);
      }
      return out;
    },

    findSailorIndexById: function (data, id) {
      for (var i = 0; i < data.sailors.length; i++) {
        if (data.sailors[i].id === id) return i;
      }
      return -1;
    },

    // Per-person: aggregate stats + a chronological list of their events.
    person: function (data, sailorIndex, fromYM, toYM) {
      var stats = emptyStats();
      var rows = [];
      var events = data.events;
      for (var i = 0; i < events.length; i++) {
        var ev = events[i];
        if (!inRange(ev, fromYM, toYM)) continue;
        var p = ev.p;
        for (var j = 0; j < p.length; j++) {
          if (p[j][0] !== sailorIndex) continue;
          var role = p[j][1];
          var st = p[j][2];
          accumulate(stats, ev, role, st);
          rows.push({
            date: ev.d,
            trip: ev.t,
            hours: ev.h,
            race: !!ev.r,
            eventId: ev.e, // MIT event id, for linking to the event page
            role: role, // "s" | "c"
            status: st, // "c" | "p" | "x" | "u"
          });
          break; // one participant record per event
        }
      }
      stats.sail_time_hrs = Math.round(stats.sail_time_hrs * 100) / 100;
      rows.sort(function (a, b) {
        return a.date < b.date ? 1 : a.date > b.date ? -1 : 0; // newest first
      });
      return { stats: stats, rows: rows };
    },

    fmt: function (value, numeric) {
      if (!numeric) return value;
      if (value == null) return "";
      if (Number.isInteger(value)) return value.toLocaleString("en-US");
      return value.toLocaleString("en-US", {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
      });
    },

    chartTextColor: function () {
      var val = getComputedStyle(document.documentElement).getPropertyValue(
        "--global-text-color"
      );
      if (val && val.trim()) return val.trim();
      // Fall back to the site's theme toggle attribute, then the OS setting.
      var theme = document.documentElement.getAttribute("data-theme");
      if (theme === "dark") return "#f2f5fa";
      if (theme === "light") return "#333";
      var dark =
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches;
      return dark ? "#f2f5fa" : "#333";
    },

    escapeHtml: function (str) {
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    },
  };

  window.BW = BW;
})();
