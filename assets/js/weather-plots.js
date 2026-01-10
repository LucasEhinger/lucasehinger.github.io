const defaultColors = [
  "#1f77b4", // 1st
  "#ff7f0e", // 2nd
  "#2ca02c", // 3rd
  "#d62728", // 4th
  "#9467bd", // 5th
  "#8c564b", // 6th
  "#e377c2", // 7th
  "#7f7f7f", // 8th
  "#bcbd22", // 9th
  "#17becf", // 10th
];

const modelMarkers = {
  hrrr: "circle",
  nam: "square",
  gfs: "diamond",
  all: "star",
};

const METERS_TO_FEET = 3.28084;
const METERS_TO_MILES = 0.000621371;
const METERS_TO_FEMTO_PARSECS = 3.24078e-2; // 1 femto-parsec ≈ 30.86 meters
const MM_TO_INCHES = 0.0393701;

function precipToUnits(valueMm, units) {
  if (valueMm == null) return null;
  if (units === "imperial") {
    return valueMm * MM_TO_INCHES;
  } else if (units === "stupid") {
    return valueMm * METERS_TO_FEMTO_PARSECS * 0.001; // mm to m to femto-parsecs
  }
  return valueMm;
}

function precipLabel(units) {
  if (units === "imperial") return "in";
  if (units === "stupid") return "fempto-pc";
  return "mm";
}

function precipRateLabel(units) {
  if (units === "imperial") return "in/hr";
  if (units === "stupid") return "fempto-pc/hr";
  return "mm/hr";
}

function visibilityToUnits(valueMeters, units) {
  if (valueMeters==null) return null;
  if (units === "imperial") {
    return valueMeters * METERS_TO_MILES;
  } else if (units === "stupid") {
    return valueMeters * METERS_TO_FEMTO_PARSECS;
  }
  return valueMeters;
}

function visibilityLabel(units) {
  if (units === "imperial") return "mi";
  if (units === "stupid") return "fempto-pc";
  return "m";
}

function getSelectedModel() {
  const selected = Array.from(
    document.querySelectorAll('input[name="model-toggle"]:checked')
  ).map((el) => el.value);
  return {
    hrrr: selected.includes("hrrr"),
    nam: selected.includes("nam"),
    gfs: selected.includes("gfs"),
  };
}

function getSelectedUnits() {
  return (
    document.querySelector('input[name="units-toggle"]:checked')?.value ||
    "metric"
  );
}

function heightToUnits(valueMeters, units) {
  if (valueMeters==null) return null;
  if (units === "imperial") {
    return valueMeters * METERS_TO_FEET;
  } else if (units === "stupid") {
    return valueMeters * METERS_TO_FEMTO_PARSECS;
  }
  return valueMeters;
}

function heightLabel(units) {
  if (units === "imperial") return "ft";
  if (units === "stupid") return "fempto-pc";
  return "m";
}

function tempLabel(units) {
  if (units === "imperial") return "F";
  if (units === "stupid") return "°R";
  return "C";
}

function chartTextColor() {
  const val = getComputedStyle(document.documentElement).getPropertyValue(
    "--text-color"
  );
  return (val && val.trim()) || "#000";
}

function axisStyle(titleText, color) {
  return {
    title: { text: titleText, font: { color } },
    tickfont: { color },
    linecolor: color,
    tickcolor: color,
    zeroline: false,
  };
}

const convertTemp = (kelvin, units) => {
  if (kelvin==null) return null;
  if (units === "imperial") {
    return (((kelvin - 273.15) * 9) / 5 + 32).toFixed(2);
  } else if (units === "stupid") {
    return ((kelvin * 9) / 5).toFixed(2);
  }
  return (kelvin - 273.15).toFixed(2);
};

function approxAltitude(meters, units) {
  const unitLabel = heightLabel(units);
  const sigRound = (val, sig = 2) => {
    if (!isFinite(val) || val === 0) return val;
    const power = sig - Math.ceil(Math.log10(Math.abs(val)));
    const factor = Math.pow(10, power);
    return Math.round(val * factor) / factor;
  };

  const raw = units === "imperial" ? meters * METERS_TO_FEET : meters;
  const value = units === "imperial" ? sigRound(raw, 2) : Math.round(raw);
  return `${value}${unitLabel}`;
}

function levelLabel(pressure, approxMeters, model, units) {
  return `${pressure}mb (~${approxAltitude(
    approxMeters,
    units
  )})<br>(${model})`;
}
function convertTimeToDateTime(timeValues, dateStr) {
  // Parse date_str like "2025-11-08 00:00" (UTC)
  const baseParts = dateStr.split(" ");
  const dateOnly = baseParts[0]; // "2025-11-08"
  const timeOnly = baseParts[1]; // "00:00"

  const [year, month, day] = dateOnly.split("-").map(Number);
  const [hour, minute] = timeOnly.split(":").map(Number);

  // Create date in UTC
  const baseDate = new Date(Date.UTC(year, month - 1, day, hour, minute, 0));

  return timeValues.map((time) => {
    const offsetDate = new Date(baseDate.getTime() + time * 60 * 60 * 1000); // time in hours

    // Convert to Eastern Time using toLocaleString
    const easternTime = offsetDate.toLocaleString("en-US", {
      timeZone: "America/New_York",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });

    // Parse the formatted string back to YYYY-MM-DD HH:mm format
    const parts = easternTime.split(", ");
    const [m, d, y] = parts[0].split("/");
    const [h, min] = parts[1].split(":");

    return `${y}-${m}-${d} ${h}:${min}`;
  });
}
function loadWeatherPlots(
  datasetId,
  modelChoice = getSelectedModel(),
  unitChoice = getSelectedUnits()
) {
  const mountainNames = [
    "MtWashington",
    "MtLafayette",
    "MtMoosilauke",
    "MtMonadnock",
  ];
  const jsonFile = `/files/weather/weather_data_${
    mountainNames[parseInt(datasetId) - 1]
  }.json`;
  const predictionsFile = "/files/weather/predictions_all.json";
  const modelSelection = modelChoice;
  const showHRRR = !!modelSelection.hrrr;
  const showNAM = !!modelSelection.nam;
  const showGFS = !!modelSelection.gfs;
  const selectedUnits = unitChoice;
  const convertHeight = (m) => heightToUnits(m, selectedUnits);
  const textColor = chartTextColor();

  Promise.all([
    fetch(jsonFile).then((response) => response.json()),
    fetch(predictionsFile).then((response) => response.json()),
  ])
    .then(([data, data_ML]) => {
      // Convert time values to full date/time using date_str
      const dateStr = data.date_str || "2025-01-01 00:00";
      console.log("date_str:", dateStr);
      console.log(
        "time values sample:",
        data.low_cloud_layer_percent_hrrr.x.slice(0, 5)
      );

      // Convert ML prediction time values
      let convertedDatesML;
      try {
        convertedDatesML = convertTimeToDateTime(
          data_ML.XGBoost_hrrr.x,
          data_ML.date_str || dateStr
        );
      } catch (err) {
        console.error("Error converting ML dates:", err);
        convertedDatesML = data_ML.XGBoost_hrrr.x;
      }

      // Update last model update time display
      const lastUpdateEl = document.getElementById("last-update");
      if (lastUpdateEl) {
        // Parse the UTC date string
        const baseParts = dateStr.split(" ");
        const dateOnly = baseParts[0];
        const timeOnly = baseParts[1];
        const [year, month, day] = dateOnly.split("-").map(Number);
        const [hour, minute] = timeOnly.split(":").map(Number);
        const utcDate = new Date(
          Date.UTC(year, month - 1, day, hour, minute, 0)
        );

        // Convert to Eastern Time
        const easternTime = utcDate.toLocaleString("en-US", {
          timeZone: "America/New_York",
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        });

        lastUpdateEl.textContent = `Last model update: ${easternTime} ET`;
      }

      let convertedDates;
      try {
        convertedDates = convertTimeToDateTime(
          data.low_cloud_layer_percent_hrrr.x,
          dateStr
        );
        console.log("converted dates sample:", convertedDates.slice(0, 5));
      } catch (err) {
        console.error("Error converting dates:", err);
        // Fallback to using raw time values if conversion fails
        convertedDates = data.low_cloud_layer_percent_hrrr.x;
      }

      // cloud_layer_percent
      const c1 = defaultColors;

      const trace_pct_low_hrrr = {
        x: convertedDates,
        y: data.low_cloud_layer_percent_hrrr.y,
        mode: "lines+markers",
        type: "scatter",
        name: "Low (HRRR)",
        line: { color: c1[0] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const trace_pct_mid_hrrr = {
        x: convertedDates,
        y: data.middle_cloud_layer_percent_hrrr.y,
        mode: "lines+markers",
        type: "scatter",
        name: "Middle (HRRR)",
        line: { color: c1[1] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const trace_pct_high_hrrr = {
        x: convertedDates,
        y: data.high_cloud_layer_percent_hrrr.y,
        mode: "lines+markers",
        type: "scatter",
        name: "High (HRRR)",
        line: { color: c1[2] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const trace_pct_boundary_hrrr = {
        x: convertedDates,
        y: data.boundary_layer_cloud_layer_hrrr.y,
        mode: "lines+markers",
        type: "scatter",
        name: "Boundary (HRRR)",
        line: { color: c1[3] },
        marker: { symbol: modelMarkers.hrrr },
      };

      const trace_pct_low_nam = {
        x: convertedDates,
        y: data.low_cloud_layer_percent_nam.y,
        mode: "lines+markers",
        line: { dash: "dash", color: c1[0] },
        type: "scatter",
        name: "Low (NAM)",
        marker: { symbol: modelMarkers.nam },
      };
      const trace_pct_mid_nam = {
        x: convertedDates,
        y: data.middle_cloud_layer_percent_nam.y,
        mode: "lines+markers",
        line: { dash: "dash", color: c1[1] },
        type: "scatter",
        name: "Middle (NAM)",
        marker: { symbol: modelMarkers.nam },
      };
      const trace_pct_high_nam = {
        x: convertedDates,
        y: data.high_cloud_layer_percent_nam.y,
        mode: "lines+markers",
        line: { dash: "dash", color: c1[2] },
        type: "scatter",
        name: "High (NAM)",
        marker: { symbol: modelMarkers.nam },
      };
      const trace_pct_boundary_nam = {
        x: convertedDates,
        y: data.boundary_layer_cloud_layer_nam.y,
        mode: "lines+markers",
        line: { dash: "dash", color: c1[3] },
        type: "scatter",
        name: "Boundary (NAM)",
        marker: { symbol: modelMarkers.nam },
      };

      const trace_pct_low_gfs = {
        x: convertedDates,
        y: data.low_cloud_layer_percent_gfs.y,
        mode: "lines+markers",
        line: { dash: "dot", color: c1[0] },
        type: "scatter",
        name: "Low (GFS)",
        marker: { symbol: modelMarkers.gfs },
      };
      const trace_pct_mid_gfs = {
        x: convertedDates,
        y: data.middle_cloud_layer_percent_gfs.y,
        mode: "lines+markers",
        line: { dash: "dot", color: c1[1] },
        type: "scatter",
        name: "Middle (GFS)",
        marker: { symbol: modelMarkers.gfs },
      };
      const trace_pct_high_gfs = {
        x: convertedDates,
        y: data.high_cloud_layer_percent_gfs.y,
        mode: "lines+markers",
        line: { dash: "dot", color: c1[2] },
        type: "scatter",
        name: "High (GFS)",
        marker: { symbol: modelMarkers.gfs },
      };
      const trace_pct_boundary_gfs = {
        x: convertedDates,
        y: data.boundary_layer_cloud_layer_gfs.y,
        mode: "lines+markers",
        line: { dash: "dot", color: c1[3] },
        type: "scatter",
        name: "Boundary (GFS)",
        marker: { symbol: modelMarkers.gfs },
      };

      const layout1 = {
        title: {
          text: "Cloud Coverage Percentage",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: { ...axisStyle("Cloud Coverage (%)", textColor) },
        legend: { font: { color: textColor } },
        showlegend: true,
      };

      const plot1Traces = [];
      if (showHRRR) {
        plot1Traces.push(
          trace_pct_low_hrrr,
          trace_pct_mid_hrrr,
          trace_pct_high_hrrr,
          trace_pct_boundary_hrrr
        );
      }
      if (showNAM) {
        plot1Traces.push(
          trace_pct_high_nam,
          trace_pct_mid_nam,
          trace_pct_low_nam,
          trace_pct_boundary_nam
        );
      }
      if (showGFS) {
        plot1Traces.push(
          trace_pct_low_gfs,
          trace_pct_mid_gfs,
          trace_pct_high_gfs,
          trace_pct_boundary_gfs
        );
      }
      Plotly.newPlot("plot1", plot1Traces, layout1);

      // Cloud ceiling + base height
      const c2 = defaultColors;

      const trace2_a = {
        x: convertedDates,
        y: data.cloud_ceiling_m_hrrr.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "Cloud<br>Ceiling (HRRR)",
        line: { color: c2[0] },
        marker: { symbol: modelMarkers.hrrr },
        showlegend: true,
      };
      const trace2_b = {
        x: convertedDates,
        y: data.cloud_base_m_hrrr.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "Cloud<br>Base (HRRR)",
        line: { color: c2[1] },
        marker: { symbol: modelMarkers.hrrr },
        showlegend: true,
      };
      const trace2_c_nam = {
        x: convertedDates,
        y: data.cloud_ceiling_nam.y.map(convertHeight),
        mode: "lines+markers",
        line: { dash: "dash", color: c2[0] },
        type: "scatter",
        name: "Cloud<br>Ceiling (NAM)",
        marker: { symbol: modelMarkers.nam },
      };
      const trace2_c = {
        x: convertedDates,
        y: data.cloud_ceiling_gfs.y.map(convertHeight),
        mode: "lines+markers",
        line: { dash: "dot", color: c2[0] },
        type: "scatter",
        name: "Cloud<br>Ceiling (GFS)",
        marker: { symbol: modelMarkers.gfs },
      };

      const layout2 = {
        title: {
          text: "Cloud Ceiling and Base Height",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: {
          ...axisStyle(`Height (${heightLabel(selectedUnits)})`, textColor),
        },
        legend: { font: { color: textColor } },
        showlegend: true,
      };

      const plot2Traces = [];
      if (showHRRR) {
        plot2Traces.push(trace2_a, trace2_b);
      }
      if (showNAM) {
        plot2Traces.push(trace2_c_nam);
      }
      if (showGFS) {
        plot2Traces.push(trace2_c);
      }
      Plotly.newPlot("plot2", plot2Traces, layout2);

      // Temperature

      const c3 = defaultColors;
      const tempUnitLabel = tempLabel(selectedUnits);

      const trace_tmp_1000mb_hrr = {
        x: convertedDates,
        y: data.tmp_1000mb_hrrr.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("1000", 100, "HRRR", selectedUnits),
        visible: true,
        line: { color: c3[0] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const tmp_925mb_hrr = {
        x: convertedDates,
        y: data.tmp_925mb_hrrr.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("925", 750, "HRRR", selectedUnits),
        line: { color: c3[1] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const tmp_850mb_hrr = {
        x: convertedDates,
        y: data.tmp_850mb_hrrr.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("850", 1500, "HRRR", selectedUnits),
        line: { color: c3[2] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const tmp_700mb_hrr = {
        x: convertedDates,
        y: data.tmp_700mb_hrrr.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("700", 3000, "HRRR", selectedUnits),
        visible: false,
        line: { color: c3[3] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const tmp_500mb_hrr = {
        x: convertedDates,
        y: data.tmp_500mb_hrrr.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("500", 5500, "HRRR", selectedUnits),
        visible: false,
        line: { color: c3[4] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const tmp_2m_hrr = {
        x: convertedDates,
        y: data.tmp_2m_hrrr.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: "2m (HRRR)",
        visible: true,
        line: { color: c3[5] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const tmp_1000mb_nam = {
        x: convertedDates,
        y: data.tmp_1000mb_nam.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("1000", 100, "NAM", selectedUnits),
        visible: true,
        line: { dash: "dash", color: c3[0] },
        marker: { symbol: modelMarkers.nam },
      };
      const tmp_925mb_nam = {
        x: convertedDates,
        y: data.tmp_925mb_nam.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("925", 750, "NAM", selectedUnits),
        line: { dash: "dash", color: c3[1] },
        marker: { symbol: modelMarkers.nam },
      };
      const tmp_850mb_nam = {
        x: convertedDates,
        y: data.tmp_850mb_nam.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("850", 1500, "NAM", selectedUnits),
        line: { dash: "dash", color: c3[2] },
        marker: { symbol: modelMarkers.nam },
      };
      const tmp_700mb_nam = {
        x: convertedDates,
        y: data.tmp_700mb_nam.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("700", 3000, "NAM", selectedUnits),
        visible: false,
        line: { dash: "dash", color: c3[3] },
        marker: { symbol: modelMarkers.nam },
      };
      const tmp_500mb_nam = {
        x: convertedDates,
        y: data.tmp_500mb_nam.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("500", 5500, "NAM", selectedUnits),
        visible: false,
        line: { dash: "dash", color: c3[4] },
        marker: { symbol: modelMarkers.nam },
      };
      const tmp_2m_nam = {
        x: convertedDates,
        y: data.tmp_2m_nam.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: "2m (NAM)",
        visible: true,
        line: { dash: "dash", color: c3[5] },
        marker: { symbol: modelMarkers.nam },
      };
      const tmp_1000mb_gfs = {
        x: convertedDates,
        y: data.tmp_1000mb_gfs.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("1000", 100, "GFS", selectedUnits),
        visible: true,
        line: { dash: "dot", color: c3[0] },
        marker: { symbol: modelMarkers.gfs },
      };
      const tmp_925mb_gfs = {
        x: convertedDates,
        y: data.tmp_925mb_gfs.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("925", 750, "GFS", selectedUnits),
        line: { dash: "dot", color: c3[1] },
        marker: { symbol: modelMarkers.gfs },
      };
      const tmp_850mb_gfs = {
        x: convertedDates,
        y: data.tmp_850mb_gfs.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("850", 1500, "GFS", selectedUnits),
        line: { dash: "dot", color: c3[2] },
        marker: { symbol: modelMarkers.gfs },
      };
      const tmp_700mb_gfs = {
        x: convertedDates,
        y: data.tmp_700mb_gfs.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("700", 3000, "GFS", selectedUnits),
        visible: false,
        line: { dash: "dot", color: c3[3] },
        marker: { symbol: modelMarkers.gfs },
      };
      const tmp_500mb_gfs = {
        x: convertedDates,
        y: data.tmp_500mb_gfs.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: levelLabel("500", 5500, "GFS", selectedUnits),
        visible: false,
        line: { dash: "dot", color: c3[4] },
        marker: { symbol: modelMarkers.gfs },
      };
      const tmp_2m_gfs = {
        x: convertedDates,
        y: data.tmp_2m_gfs.y.map((v) => convertTemp(v, selectedUnits)),
        mode: "lines+markers",
        type: "scatter",
        name: "2m (GFS)",
        visible: true,
        line: { dash: "dot", color: c3[5] },
        marker: { symbol: modelMarkers.gfs },
      };
      const layout3 = {
        title: {
          text: "Temperature at Various Altitudes",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: { ...axisStyle(`Temperature (${tempUnitLabel})`, textColor) },
        legend: { font: { color: textColor } },
        showlegend: true,
      };
      const plot3Traces = [];
      if (showHRRR) {
        plot3Traces.push(
          trace_tmp_1000mb_hrr,
          tmp_925mb_hrr,
          tmp_850mb_hrr,
          tmp_700mb_hrr,
          tmp_500mb_hrr,
          tmp_2m_hrr
        );
      }
      if (showNAM) {
        plot3Traces.push(
          tmp_1000mb_nam,
          tmp_925mb_nam,
          tmp_850mb_nam,
          tmp_700mb_nam,
          tmp_500mb_nam,
          tmp_2m_nam
        );
      }
      if (showGFS) {
        plot3Traces.push(
          tmp_1000mb_gfs,
          tmp_925mb_gfs,
          tmp_850mb_gfs,
          tmp_700mb_gfs,
          tmp_500mb_gfs,
          tmp_2m_gfs
        );
      }
      Plotly.newPlot("plot3", plot3Traces, layout3);

      const c4 = defaultColors;

      const trace_hpbl_hrrr = {
        x: convertedDates,
        y: data.hpbl_surface_hrrr.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "HPBL (HRRR)",
        line: { color: c4[0] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const trace_hpbl_nam = {
        x: convertedDates,
        y: data.hpbl_surface_nam.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "HPBL (NAM)",
        line: { dash: "dash", color: c4[0] },
        marker: { symbol: modelMarkers.nam },
      };
      const trace_bpbl_gfs = {
        x: convertedDates,
        y: data.hpbl_surface_gfs.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "BPBL (GFS)",
        line: { dash: "dot", color: c4[1] },
        marker: { symbol: modelMarkers.gfs },
      };
      const layout4 = {
        title: {
          text: "Planetary Boundary Layer Height",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: {
          ...axisStyle(`Height (${heightLabel(selectedUnits)})`, textColor),
        },
        legend: { font: { color: textColor } },
        showlegend: true,
      };
      const plot4Traces = [];
      if (showHRRR) {
        plot4Traces.push(trace_hpbl_hrrr);
      }
      if (showNAM) {
        plot4Traces.push(trace_hpbl_nam);
      }
      if (showGFS) {
        plot4Traces.push(trace_bpbl_gfs);
      }
      Plotly.newPlot("plot4", plot4Traces, layout4);

      const c5 = defaultColors;

      const trace5_a = {
        x: convertedDates,
        y: data.rh_2m_hrrr.y,
        mode: "lines+markers",
        type: "scatter",
        name: "2m RH (HRRR)",
        line: { color: c5[0] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const trace5_b_nam = {
        x: convertedDates,
        y: data.rh_2m_nam.y,
        mode: "lines+markers",
        type: "scatter",
        name: "2m RH (NAM)",
        line: { dash: "dash", color: c5[0] },
        marker: { symbol: modelMarkers.nam },
      };
      const trace5_c_nam = {
        x: convertedDates,
        y: data.rh_925mb_nam.y,
        mode: "lines+markers",
        type: "scatter",
        name: "925mb RH (NAM)",
        line: { dash: "dash", color: c5[1] },
        marker: { symbol: modelMarkers.nam },
      };
      const trace5_b = {
        x: convertedDates,
        y: data.rh_2m_gfs.y,
        mode: "lines+markers",
        type: "scatter",
        name: "2m RH (GFS)",
        line: { dash: "dot", color: c5[0] },
        marker: { symbol: modelMarkers.gfs },
      };
      const trace5_c = {
        x: convertedDates,
        y: data.rh_925mb_gfs.y,
        mode: "lines+markers",
        type: "scatter",
        name: "925mb RH (GFS)",
        line: { dash: "dot", color: c5[1] },
        marker: { symbol: modelMarkers.gfs },
      };
      const layout5 = {
        title: {
          text: "Relative Humidity",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: { ...axisStyle("Relative Humidity (%)", textColor) },
        legend: { font: { color: textColor } },
        showlegend: true,
      };
      const plot5Traces = [];
      if (showHRRR) {
        plot5Traces.push(trace5_a);
      }
      if (showNAM) {
        plot5Traces.push(trace5_b_nam, trace5_c_nam);
      }
      if (showGFS) {
        plot5Traces.push(trace5_b, trace5_c);
      }
      Plotly.newPlot("plot5", plot5Traces, layout5);

      const c6 = defaultColors;

      const trace6_a = {
        x: convertedDates,
        y: data.hgt_0C_iso_hrrr.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "0°C Isotherm<br>Height (HRRR)",
        line: { color: c6[0] },
        marker: { symbol: modelMarkers.hrrr },
      };
      const trace6_b_nam = {
        x: convertedDates,
        y: data.hgt_0C_iso_nam.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "0°C Isotherm<br>Height (NAM)",
        line: { dash: "dash", color: c6[0] },
        marker: { symbol: modelMarkers.nam },
      };
      const trace6_b = {
        x: convertedDates,
        y: data.hgt_0C_iso_gfs.y.map(convertHeight),
        mode: "lines+markers",
        type: "scatter",
        name: "0°C Isotherm<br>Height (GFS)",
        line: { dash: "dot", color: c6[0] },
        marker: { symbol: modelMarkers.gfs },
      };
      const layout6 = {
        title: {
          text: "0°C Isotherm Height",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: {
          ...axisStyle(`Height (${heightLabel(selectedUnits)})`, textColor),
        },
        legend: { font: { color: textColor } },
        showlegend: true,
      };
      const plot6Traces = [];
      if (showHRRR) {
        plot6Traces.push(trace6_a);
      }
      if (showNAM) {
        plot6Traces.push(trace6_b_nam);
      }
      if (showGFS) {
        plot6Traces.push(trace6_b);
      }
      Plotly.newPlot("plot6", plot6Traces, layout6);

      const c7 = defaultColors;

      const convertVisibility = (m) => visibilityToUnits(m, selectedUnits);

      const trace7_a = {
        x: convertedDates,
        y: data.vis_surface_hrrr.y.map(convertVisibility),
        mode: "lines+markers",
        type: "scatter",
        name: "Surface<br>Visibility (HRRR)",
        line: { color: c7[0] },
        marker: { symbol: modelMarkers.hrrr },
      };

      const trace7_b_nam = {
        x: convertedDates,
        y: data.vis_surface_nam.y.map(convertVisibility),
        mode: "lines+markers",
        type: "scatter",
        name: "Surface<br>Visibility (NAM)",
        line: { dash: "dash", color: c7[0] },
        marker: { symbol: modelMarkers.nam },
      };

      const trace7_b = {
        x: convertedDates,
        y: data.vis_surface_gfs.y.map(convertVisibility),
        mode: "lines+markers",
        type: "scatter",
        name: "Surface<br>Visibility (GFS)",
        line: { dash: "dot", color: c7[0] },
        marker: { symbol: modelMarkers.gfs },
      };

      const layout7 = {
        title: {
          text: "Surface<br>Visibility",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: {
          ...axisStyle(
            `Visibility (${visibilityLabel(selectedUnits)})`,
            textColor
          ),
        },
        legend: { font: { color: textColor } },
        showlegend: true,
      };
      const plot7Traces = [];
      if (showHRRR) {
        plot7Traces.push(trace7_a);
      }
      if (showNAM) {
        plot7Traces.push(trace7_b_nam);
      }
      if (showGFS) {
        plot7Traces.push(trace7_b);
      }
      Plotly.newPlot("plot7", plot7Traces, layout7);

      // Plot 8: Undercast Probability
      const c8 = defaultColors;
      const mlColors = {
        xgb: c8[0],
        rf: c8[1],
        gbdt: c8[2],
        consensus: c8[3],
      };

      const trace8_xgb_hrrr = {
        x: convertedDatesML,
        y: data_ML.XGBoost_hrrr.y,
        mode: "lines+markers",
        type: "scatter",
        name: "HRRR (XGBoost)",
        line: { color: mlColors.xgb },
        marker: { symbol: modelMarkers.hrrr },
      };

      const trace8_rf_hrrr = {
        x: convertedDatesML,
        y: data_ML["Random Forest_hrrr"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "HRRR (Random Forest)",
        line: { dash: "dash", color: mlColors.rf },
        marker: { symbol: modelMarkers.hrrr },
      };

      const trace8_gbdt_hrrr = {
        x: convertedDatesML,
        y: data_ML["Gradient Boosting_hrrr"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "HRRR (Gradient Boosting)",
        line: { dash: "dot", color: mlColors.gbdt },
        marker: { symbol: modelMarkers.hrrr },
      };

      const trace8_consensus_hrrr = {
        x: convertedDatesML,
        y: data_ML.consensus_hrrr.y,
        mode: "lines+markers",
        type: "scatter",
        name: "HRRR (Consensus)",
        line: { dash: "longdash", color: mlColors.consensus },
        marker: { symbol: modelMarkers.hrrr },
      };

      const trace8_xgb_nam = {
        x: convertedDatesML,
        y: data_ML.XGBoost_nam.y,
        mode: "lines+markers",
        type: "scatter",
        name: "NAM (XGBoost)",
        line: { color: mlColors.xgb },
        marker: { symbol: modelMarkers.nam },
      };

      const trace8_rf_nam = {
        x: convertedDatesML,
        y: data_ML["Random Forest_nam"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "NAM (Random Forest)",
        line: { dash: "dash", color: mlColors.rf },
        marker: { symbol: modelMarkers.nam },
      };

      const trace8_gbdt_nam = {
        x: convertedDatesML,
        y: data_ML["Gradient Boosting_nam"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "NAM (Gradient Boosting)",
        line: { dash: "dot", color: mlColors.gbdt },
        marker: { symbol: modelMarkers.nam },
      };

      const trace8_consensus_nam = {
        x: convertedDatesML,
        y: data_ML.consensus_nam.y,
        mode: "lines+markers",
        type: "scatter",
        name: "NAM (Consensus)",
        line: { dash: "longdash", color: mlColors.consensus },
        marker: { symbol: modelMarkers.nam },
      };

      const trace8_xgb_gfs = {
        x: convertedDatesML,
        y: data_ML.XGBoost_gfs.y,
        mode: "lines+markers",
        type: "scatter",
        name: "GFS (XGBoost)",
        line: { color: mlColors.xgb },
        marker: { symbol: modelMarkers.gfs },
      };

      const trace8_rf_gfs = {
        x: convertedDatesML,
        y: data_ML["Random Forest_gfs"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "GFS (Random Forest)",
        line: { dash: "dash", color: mlColors.rf },
        marker: { symbol: modelMarkers.gfs },
      };

      const trace8_gbdt_gfs = {
        x: convertedDatesML,
        y: data_ML["Gradient Boosting_gfs"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "GFS (Gradient Boosting)",
        line: { dash: "dot", color: mlColors.gbdt },
        marker: { symbol: modelMarkers.gfs },
      };

      const trace8_consensus_gfs = {
        x: convertedDatesML,
        y: data_ML.consensus_gfs.y,
        mode: "lines+markers",
        type: "scatter",
        name: "GFS (Consensus)",
        line: { dash: "longdash", color: mlColors.consensus },
        marker: { symbol: modelMarkers.gfs },
      };

      const trace8_xgb_all = {
        x: convertedDatesML,
        y: data_ML.XGBoost_all.y,
        mode: "lines+markers",
        type: "scatter",
        name: "All (XGBoost)",
        line: { color: mlColors.xgb },
        marker: { symbol: modelMarkers.all },
      };

      const trace8_rf_all = {
        x: convertedDatesML,
        y: data_ML["Random Forest_all"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "All (Random Forest)",
        line: { dash: "dash", color: mlColors.rf },
        marker: { symbol: modelMarkers.all },
      };

      const trace8_gbdt_all = {
        x: convertedDatesML,
        y: data_ML["Gradient Boosting_all"].y,
        mode: "lines+markers",
        type: "scatter",
        name: "All (Gradient Boosting)",
        line: { dash: "dot", color: mlColors.gbdt },
        marker: { symbol: modelMarkers.all },
      };

      const trace8_consensus_all = {
        x: convertedDatesML,
        y: data_ML.consensus_all.y,
        mode: "lines+markers",
        type: "scatter",
        name: "All (Consensus)",
        line: { dash: "longdash", color: mlColors.consensus },
        marker: { symbol: modelMarkers.all },
      };

      const layout8 = {
        title: {
          text: "Undercast Prediction",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: {
          ...axisStyle("", textColor),
          range: [-0.5, 1.5],
          tickmode: "array",
          tickvals: [1, 0],
          ticktext: ["Undercast", "Not Undercast"],
          tickangle: 90,
        },
        legend: { font: { color: textColor } },
        showlegend: true,
      };

      const plot8Traces = [];
      if (showHRRR) {
        plot8Traces.push(
          trace8_consensus_hrrr
        );
      }
      if (showNAM) {
        plot8Traces.push(
          trace8_consensus_nam
        );
      }
      if (showGFS) {
        plot8Traces.push(
          trace8_consensus_gfs
        );
      }
      plot8Traces.push(
        trace8_consensus_all
      );
      Plotly.newPlot("plot8", plot8Traces, layout8);

      // Plot 9: Undercast Probability (Other Models)
      const layout9 = {
        title: {
          text: "Undercast Prediction (Other Models)",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: {
          ...axisStyle("", textColor),
          range: [-0.5, 1.5],
          tickmode: "array",
          tickvals: [1, 0],
          ticktext: ["Undercast", "Not Undercast"],
          tickangle: 90,
        },
        legend: { font: { color: textColor } },
        showlegend: true,
      };

      const plot9Traces = [];
      if (showHRRR) {
        plot9Traces.push(trace8_xgb_hrrr, trace8_rf_hrrr, trace8_gbdt_hrrr);
      }
      if (showNAM) {
        plot9Traces.push(trace8_xgb_nam, trace8_rf_nam, trace8_gbdt_nam);
      }
      if (showGFS) {
        plot9Traces.push(trace8_xgb_gfs, trace8_rf_gfs, trace8_gbdt_gfs);
      }
      plot9Traces.push(trace8_xgb_all, trace8_rf_all, trace8_gbdt_all);
      Plotly.newPlot("plot9", plot9Traces, layout9);

      // Plot 10: Precipitation
      const c9 = defaultColors;

      const convertPrecip = (mm) => precipToUnits(mm, selectedUnits);
      const convertPrecipRate = (mmPerSec) => {
        if (mmPerSec == null) return null;
        const mmPerHr = mmPerSec * 3600; // Convert mm/s to mm/hr
        return precipToUnits(mmPerHr, selectedUnits);
      };

      const trace10_apcp_hrrr = {
        x: convertedDates,
        y: data.apcp_surface_hrrr.y.map(convertPrecip),
        mode: "lines+markers",
        type: "scatter",
        name: "Accumulated Precip (HRRR)",
        line: { color: c9[0] },
        marker: { symbol: modelMarkers.hrrr },
        yaxis: "y",
      };

      const trace10_prate_hrrr = {
        x: convertedDates,
        y: data.prate_surface_hrrr.y.map(convertPrecipRate),
        mode: "lines+markers",
        type: "scatter",
        name: "Precip Rate (HRRR)",
        line: { color: c9[1] },
        marker: { symbol: modelMarkers.hrrr },
        yaxis: "y2",
      };

      const trace10_apcp_nam = {
        x: convertedDates,
        y: data.apcp_surface_nam.y.map(convertPrecip),
        mode: "lines+markers",
        type: "scatter",
        name: "Accumulated Precip (NAM)",
        line: { dash: "dash", color: c9[0] },
        marker: { symbol: modelMarkers.nam },
        yaxis: "y",
      };

      const trace10_prate_nam = {
        x: convertedDates,
        y: data.prate_surface_nam.y.map(convertPrecipRate),
        mode: "lines+markers",
        type: "scatter",
        name: "Precip Rate (NAM)",
        line: { dash: "dash", color: c9[1] },
        marker: { symbol: modelMarkers.nam },
        yaxis: "y2",
      };

      const trace10_apcp_gfs = {
        x: convertedDates,
        y: data.apcp_surface_gfs.y.map(convertPrecip),
        mode: "lines+markers",
        type: "scatter",
        name: "Accumulated Precip (GFS)",
        line: { dash: "dot", color: c9[0] },
        marker: { symbol: modelMarkers.gfs },
        yaxis: "y",
      };

      const trace10_prate_gfs = {
        x: convertedDates,
        y: data.prate_surface_gfs.y.map(convertPrecipRate),
        mode: "lines+markers",
        type: "scatter",
        name: "Precip Rate (GFS)",
        line: { dash: "dot", color: c9[1] },
        marker: { symbol: modelMarkers.gfs },
        yaxis: "y2",
      };

      const layout10 = {
        title: {
          text: "Precipitation",
          font: { color: textColor },
        },
        xaxis: { ...axisStyle("", textColor) },
        yaxis: { 
          ...axisStyle(`Accumulated (${precipLabel(selectedUnits)})`, textColor),
          rangemode: "tozero",
        },
        yaxis2: {
          ...axisStyle(`Rate (${precipRateLabel(selectedUnits)})`, textColor),
          overlaying: "y",
          side: "right",
          rangemode: "tozero",
          matches: "y",
        },
        legend: { font: { color: textColor } },
        showlegend: true,
      };

      const plot10Traces = [];
      if (showHRRR) {
        plot10Traces.push(trace10_apcp_hrrr, trace10_prate_hrrr);
      }
      if (showNAM) {
        plot10Traces.push(trace10_apcp_nam, trace10_prate_nam);
      }
      if (showGFS) {
        plot10Traces.push(trace10_apcp_gfs, trace10_prate_gfs);
      }
      
      Plotly.newPlot("plot10", plot10Traces, layout10);

      // Add tooltips to plot info icons after Plotly renders
      setTimeout(() => {
        attachPlotInfoTooltips();
      }, 100);
    })
    .catch((error) => console.error("Error loading weather data:", error));
}

function attachPlotInfoTooltips() {
  const unit = getSelectedUnits();
  const isImperial = unit === "imperial";
  const useKm = unit !== "imperial"; // metric and stupid both use km per request

  const plot1Desc = `Cloud coverage percentage for\n- High (${
    isImperial ? "20,000–43,000 ft" : "6–13 km"
  })\n- Middle (${isImperial ? "6,500–20,000 ft" : "2–6 km"})\n- Low (${
    isImperial ? "1,500–6,500 ft" : "0.5–2 km"
  })\n- Boundary Layer (${isImperial ? "0–6500 ft" : "0–2 km"})`;

  const plotInfoMap = {
    plot1: plot1Desc,
    plot2:
      "Cloud base: the lowest height above ground level (AGL) at which cloud cover exists.\nCloud height: the lowest height above ground level (AGL) at which >60% cloud cover exists.",
    plot3:
      "Temperature at different pressure levels (altitudes) and at 2m above ground",
    plot4:
      "The height of the atmospheric boundary layer where surface effects dominate",
    plot5: "Relative humidity at 2m above ground and at the 925mb level",
    plot6:
      "Elevation at which temperature reaches 0°C (32°F) - important for rain/snow line. If 0, then freezing point is at sea level.",
    plot7:
      "Horizontal visibility at surface level - affected by fog, precipitation, and haze.",
    plot8:
      "Probability of undercast conditions - when clouds form below summit elevation.",
    plot9:
      `Accumulated precipitation and precipitation rate at the surface. Units adjust based on selection (${
        unit === "imperial" ? "inches" : unit === "stupid" ? "fempto-parsecs" : "millimeters"
      }).`,
  };

  Object.entries(plotInfoMap).forEach(([plotId, tooltipText]) => {
    const plotDiv = document.getElementById(plotId);
    if (!plotDiv) return;

    // Remove old button if exists
    const oldButton = plotDiv.querySelector(".plot-info-button");
    if (oldButton) oldButton.remove();

    // Remove old tooltip if exists
    const oldTooltip = plotDiv.querySelector(".plot-info-tooltip");
    if (oldTooltip) oldTooltip.remove();

    // Ensure plot div is positioned relatively
    plotDiv.style.position = "relative";

    // Create info button at top-left
    const button = document.createElement("div");
    button.className = "plot-info-button";
    button.textContent = "ⓘ";
    button.style.position = "absolute";
    button.style.top = "8px";
    button.style.left = "8px";
    button.style.zIndex = "10";
    plotDiv.appendChild(button);

    // Create tooltip
    const tooltip = document.createElement("div");
    tooltip.className = "plot-info-tooltip";
    tooltip.textContent = tooltipText;
    tooltip.style.position = "absolute";
    tooltip.style.top = "30px";
    tooltip.style.left = "8px";
    tooltip.style.display = "none";
    plotDiv.appendChild(tooltip);

    // Show/hide tooltip on button hover
    button.addEventListener("mouseenter", () => {
      tooltip.style.display = "block";
    });
    button.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });
    tooltip.addEventListener("mouseenter", () => {
      tooltip.style.display = "block";
    });
    tooltip.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });
  });
}

document.getElementById("weather-toggle").addEventListener("change", (e) => {
  loadWeatherPlots(e.target.value, getSelectedModel(), getSelectedUnits());
  setTimeout(attachPlotInfoTooltips, 150);
});

document.querySelectorAll('input[name="model-toggle"]').forEach((input) => {
  input.addEventListener("change", () => {
    const datasetId = document.getElementById("weather-toggle").value;
    loadWeatherPlots(datasetId, getSelectedModel(), getSelectedUnits());
    setTimeout(attachPlotInfoTooltips, 150);
  });
});

document.querySelectorAll('input[name="units-toggle"]').forEach((input) => {
  input.addEventListener("change", () => {
    const datasetId = document.getElementById("weather-toggle").value;
    loadWeatherPlots(datasetId, getSelectedModel(), getSelectedUnits());
    setTimeout(attachPlotInfoTooltips, 150);
  });
});

loadWeatherPlots("1", getSelectedModel(), getSelectedUnits());
