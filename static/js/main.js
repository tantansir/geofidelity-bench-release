/* GeoFidelity-Bench project page.
   All numbers below are copied from the camera-ready paper (NeurIPS 2026). */
(function () {
  "use strict";

  // ------------------------------------------------------------------ data

  var MODELS = [
    { id: "sdxl", name: "SDXL", L0: 0.512, L1: 0.536, L2: 0.536 },
    { id: "sd35_large", name: "SD 3.5 Large", L0: 0.432, L1: 0.515, L2: 0.516 },
    { id: "flux_dev", name: "FLUX.1-dev", L0: 0.456, L1: 0.484, L2: 0.496 },
    { id: "flux_schnell", name: "FLUX.1-schnell", L0: 0.465, L1: 0.499, L2: 0.507 },
    { id: "pixart_sigma", name: "PixArt-Σ", L0: 0.459, L1: 0.486, L2: 0.489 },
    { id: "hunyuan_dit", name: "HunyuanDiT", L0: 0.458, L1: 0.477, L2: 0.478 }
  ];

  var HIERARCHY = [
    { key: "Same block", short: "Same block", real: 0.734, gen: 0.451, target: true },
    { key: "Same city (1)", short: "Same city 1", real: 0.557, gen: 0.449 },
    { key: "Same city (2)", short: "Same city 2", real: 0.526, gen: 0.426 },
    { key: "Same driving side", short: "Same drive", real: 0.193, gen: 0.143 },
    { key: "Random city", short: "Random city", real: 0.204, gen: 0.150 }
  ];

  var CONTROLS = [
    { name: "Wrong street", short: "Wrong street", cos: [0.001, -0.004, 0.006], ret: [0.025, 0.008, 0.042], cosAbs: 0.498, retAbs: 0.373 },
    { name: "Shuffled neighborhood", short: "Shuffled nbhd.", cos: [0.007, 0.003, 0.012], ret: [0.038, 0.020, 0.057], cosAbs: 0.492, retAbs: 0.361 },
    { name: "Wrong street and neighborhood", short: "Both wrong", cos: [0.011, 0.005, 0.017], ret: [0.054, 0.034, 0.074], cosAbs: 0.488, retAbs: 0.344 }
  ];

  var CLIP_RHO = [
    { name: "Target CosSim", exact: [0.112, -0.008, 0.231], loc: [0.238, 0.114, 0.353] },
    { name: "Local margin", exact: [0.005, -0.117, 0.126], loc: [0.123, -0.012, 0.249] },
    { name: "Retrieval accuracy", exact: [0.041, -0.068, 0.149], loc: [0.105, -0.019, 0.222] }
  ];
  var CLIP_AUC = { name: "All four retrieve", exact: [0.508, 0.442, 0.572], loc: [0.579, 0.505, 0.652] };

  var QUARTILE = [
    { name: "Lowest CLIP quartile", v: 53.0 },
    { name: "All 672 panels", v: 47.3 },
    { name: "Highest CLIP quartile", v: 46.4 }
  ];

  var BOARD = {
    gens: [
      { name: "SDXL", cos: 0.536, dcsf: 0.547, mmd: 0.457, gaas: 0.380, ret: 0.408 },
      { name: "SD 3.5 Large", cos: 0.515, dcsf: 0.565, mmd: 0.470, gaas: 0.387, ret: 0.375 },
      { name: "FLUX.1-schnell", cos: 0.499, dcsf: 0.571, mmd: 0.481, gaas: 0.395, ret: 0.382 },
      { name: "PixArt-Σ", cos: 0.486, dcsf: 0.625, mmd: 0.522, gaas: 0.401, ret: 0.422 },
      { name: "FLUX.1-dev", cos: 0.484, dcsf: 0.634, mmd: 0.527, gaas: 0.402, ret: 0.388 },
      { name: "HunyuanDiT", cos: 0.477, dcsf: 0.637, mmd: 0.531, gaas: 0.421, ret: 0.420 }
    ],
    mean: { name: "Six-model mean", cos: 0.499, dcsf: 0.596, mmd: 0.498, gaas: 0.398, ret: 0.399 },
    anchors: [
      { name: "Held-out Real", cos: 0.902, dcsf: 0.036, mmd: 0.020, gaas: 0.285, ret: 0.855, same: true },
      { name: "Random-Same-Country", cos: 0.710, dcsf: 0.160, mmd: 0.151, gaas: 0.365, ret: 0.259 },
      { name: "Random-Global", cos: 0.333, dcsf: 0.249, mmd: 0.249, gaas: 0.370, ret: 0.163 }
    ]
  };
  var BOARD_COLS = [
    { key: "cos", higher: true },
    { key: "dcsf", higher: false },
    { key: "mmd", higher: false },
    { key: "gaas", higher: false },
    { key: "ret", higher: true }
  ];

  // name, ISO code, country, lat, lon, driving side, blocks, reference assignments, metro code
  var CITIES = [
    ["Amsterdam", "NL", "Netherlands", 52.37, 4.90, "R", 6, 398, "AMS"],
    ["Bangkok", "TH", "Thailand", 13.76, 100.50, "L", 3, 170, "BKK"],
    ["Berlin", "DE", "Germany", 52.52, 13.40, "R", 1, 28, "BER"],
    ["Bogota", "CO", "Colombia", 4.71, -74.07, "R", 2, 153, "BOG"],
    ["Buenos Aires", "AR", "Argentina", -34.60, -58.38, "R", 6, 376, "BUE"],
    ["Cairo", "EG", "Egypt", 30.04, 31.24, "R", 2, 100, "CAI"],
    ["Cape Town", "ZA", "South Africa", -33.92, 18.42, "L", 3, 189, "CPT"],
    ["Dubai", "AE", "United Arab Emirates", 25.20, 55.27, "R", 7, 445, "DXB"],
    ["Istanbul", "TR", "Türkiye", 41.01, 28.98, "R", 5, 300, "IST"],
    ["London", "GB", "United Kingdom", 51.51, -0.13, "L", 7, 445, "LON"],
    ["Melbourne", "AU", "Australia", -37.81, 144.96, "L", 1, 68, "MEL"],
    ["Mexico City", "MX", "Mexico", 19.43, -99.13, "R", 2, 151, "MEX"],
    ["Mumbai", "IN", "India", 19.08, 72.88, "L", 6, 403, "BOM"],
    ["Nairobi", "KE", "Kenya", -1.29, 36.82, "L", 1, 38, "NBO"],
    ["New York", "US", "United States", 40.71, -74.01, "R", 9, 767, "NYC"],
    ["Paris", "FR", "France", 48.86, 2.35, "R", 8, 500, "PAR"],
    ["Rome", "IT", "Italy", 41.90, 12.50, "R", 5, 382, "ROM"],
    ["San Francisco", "US", "United States", 37.77, -122.42, "R", 2, 66, "SFO"],
    ["Sao Paulo", "BR", "Brazil", -23.55, -46.63, "R", 3, 142, "SAO"],
    ["Seoul", "KR", "South Korea", 37.57, 126.98, "R", 5, 311, "SEL"],
    ["Shanghai", "CN", "China", 31.23, 121.47, "R", 4, 334, "SHA"],
    ["Singapore", "SG", "Singapore", 1.35, 103.82, "L", 3, 182, "SIN"],
    ["Sydney", "AU", "Australia", -33.87, 151.21, "L", 4, 213, "SYD"],
    ["Tokyo", "JP", "Japan", 35.68, 139.65, "L", 10, 910, "TYO"],
    ["Toronto", "CA", "Canada", 43.65, -79.38, "R", 7, 492, "YTO"]
  ].map(function (r) {
    return { name: r[0], cc: r[1], country: r[2], lat: r[3], lon: r[4], drive: r[5], blocks: r[6], refs: r[7], code: r[8] };
  });

  // Cross-city DINOv2 similarity (Appendix C), rows and columns in CITIES order.
  var CROSS = [
    [1.00,.11,.50,.04,.09,.24,.27,.17,.20,.40,.00,.05,.17,.16,.21,.24,.29,.06,.03,.20,.24,.15,.15,.10,.18],
    [.11,1.00,.09,.31,.35,.42,.32,.24,.33,.07,.18,.33,.51,.57,.20,.07,.31,.05,.36,.23,.29,.60,.23,.07,.28],
    [.50,.09,1.00,.04,.15,.22,.26,.21,.27,.31,-.02,.07,.10,.08,.16,.34,.37,-.01,.00,.37,.24,.11,.12,.10,.20],
    [.04,.31,.04,1.00,.78,.19,.70,.18,.24,.17,.54,.80,.24,.32,.21,.12,.21,.44,.64,.33,.23,.67,.45,.13,.19],
    [.09,.35,.15,.78,1.00,.29,.76,.20,.34,.20,.54,.76,.20,.24,.34,.28,.29,.23,.68,.35,.29,.66,.48,.06,.27],
    [.24,.42,.22,.19,.29,1.00,.40,.51,.65,.16,.10,.21,.49,.58,.39,.25,.66,.17,.21,.29,.38,.38,.30,.05,.46],
    [.27,.32,.26,.70,.76,.40,1.00,.30,.42,.29,.51,.72,.28,.31,.32,.29,.41,.39,.58,.43,.36,.70,.56,.15,.28],
    [.17,.24,.21,.18,.20,.51,.30,1.00,.28,.14,.07,.18,.34,.38,.10,.15,.29,.10,.09,.32,.25,.26,.12,.02,.20],
    [.20,.33,.27,.24,.34,.65,.42,.28,1.00,.25,.18,.26,.36,.46,.35,.23,.70,.25,.23,.25,.33,.33,.49,.11,.43],
    [.40,.07,.31,.17,.20,.16,.29,.14,.25,1.00,.11,.16,.16,.21,.22,.32,.23,.08,.10,.34,.20,.21,.24,.08,.15],
    [.00,.18,-.02,.54,.54,.10,.51,.07,.18,.11,1.00,.60,.08,.16,.08,.05,.10,.30,.68,.16,.11,.50,.47,.06,.11],
    [.05,.33,.07,.80,.76,.21,.72,.18,.26,.16,.60,1.00,.28,.31,.17,.15,.20,.34,.71,.33,.28,.67,.45,.11,.13],
    [.17,.51,.10,.24,.20,.49,.28,.34,.36,.16,.08,.28,1.00,.70,.16,.10,.42,.11,.32,.30,.53,.53,.20,.10,.17],
    [.16,.57,.08,.32,.24,.58,.31,.38,.46,.21,.16,.31,.70,1.00,.26,.06,.51,.19,.31,.24,.44,.54,.30,.12,.41],
    [.21,.20,.16,.21,.34,.39,.32,.10,.35,.22,.08,.17,.16,.26,1.00,.29,.42,.34,.28,.29,.34,.25,.36,.04,.67],
    [.24,.07,.34,.12,.28,.25,.29,.15,.23,.32,.05,.15,.10,.06,.29,1.00,.34,.05,.15,.26,.32,.17,.18,.08,.14],
    [.29,.31,.37,.21,.29,.66,.41,.29,.70,.23,.10,.20,.42,.51,.42,.34,1.00,.26,.23,.30,.46,.33,.29,.15,.48],
    [.06,.05,-.01,.44,.23,.17,.39,.10,.25,.08,.30,.34,.11,.19,.34,.05,.26,1.00,.28,.17,.16,.27,.40,.19,.34],
    [.03,.36,.00,.64,.68,.21,.58,.09,.23,.10,.68,.71,.32,.31,.28,.15,.23,.28,1.00,.30,.39,.73,.43,.09,.18],
    [.20,.23,.37,.33,.35,.29,.43,.32,.25,.34,.16,.33,.30,.24,.29,.26,.30,.17,.30,1.00,.50,.37,.28,.36,.25],
    [.24,.29,.24,.23,.29,.38,.36,.25,.33,.20,.11,.28,.53,.44,.34,.32,.46,.16,.39,.50,1.00,.45,.27,.24,.24],
    [.15,.60,.11,.67,.66,.38,.70,.26,.33,.21,.50,.67,.53,.54,.25,.17,.33,.27,.73,.37,.45,1.00,.45,.10,.25],
    [.15,.23,.12,.45,.48,.30,.56,.12,.49,.24,.47,.45,.20,.30,.36,.18,.29,.40,.43,.28,.27,.45,1.00,.11,.35],
    [.10,.07,.10,.13,.06,.05,.15,.02,.11,.08,.06,.11,.10,.12,.04,.08,.15,.19,.09,.36,.24,.10,.11,1.00,.05],
    [.18,.28,.20,.19,.27,.46,.28,.20,.43,.15,.11,.13,.17,.41,.67,.14,.48,.34,.18,.25,.24,.25,.35,.05,1.00]
  ];
  // Average-linkage order on 1 - similarity, used by the "Clustered" toggle.
  var CROSS_CLUSTER = [23, 9, 0, 2, 15, 20, 19, 7, 8, 16, 5, 12, 13, 1, 21, 18, 11, 3, 4, 6, 10, 22, 17, 14, 24];

  var QUAL_CITIES = {
    paris: "Paris, FR", tokyo: "Tokyo, JP", new_york: "New York, US",
    cairo: "Cairo, EG", bangkok: "Bangkok, TH", buenos_aires: "Buenos Aires, AR"
  };
  var QUAL_MODELS = [
    ["sdxl", "SDXL"], ["sd35_large", "SD 3.5 Large"], ["flux_dev", "FLUX.1-dev"],
    ["flux_schnell", "FLUX.1-schnell"], ["pixart_sigma", "PixArt-Σ"], ["hunyuan_dit", "HunyuanDiT"]
  ];

  // ------------------------------------------------------------ utilities

  var SVGNS = "http://www.w3.org/2000/svg";
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function S(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function T(parent, x, y, text, cls, attrs) {
    var t = S("text", Object.assign({ x: x, y: y, "class": cls || "" }, attrs || {}), parent);
    t.textContent = text;
    return t;
  }
  function lin(d0, d1, r0, r1) {
    var f = function (v) { return r0 + (v - d0) * (r1 - r0) / (d1 - d0); };
    return f;
  }
  function f3(v) { return v.toFixed(3); }
  function sgn(v, d) {
    d = d === undefined ? 3 : d;
    var s = Math.abs(v).toFixed(d);
    if (Number(s) === 0) return s;
    return (v > 0 ? "+" : "−") + s;
  }
  function fmtTick(v, d) {
    var s = Math.abs(v).toFixed(d);
    return (v < 0 && Number(s) !== 0 ? "−" : "") + s;
  }
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  var measureSvg = null;
  function textWidth(text, cls) {
    if (!measureSvg) {
      measureSvg = S("svg", { width: 0, height: 0, "aria-hidden": "true", style: "position:absolute;left:-9999px;top:-9999px" });
      document.body.appendChild(measureSvg);
    }
    var t = T(measureSvg, 0, 0, text, cls);
    var w = t.getComputedTextLength();
    measureSvg.removeChild(t);
    return w;
  }

  var charts = [];
  function mount(container, draw) {
    var last = 0;
    function render(force) {
      var w = Math.floor(container.clientWidth);
      if (!w || (w === last && !force)) return;
      last = w;
      while (container.firstChild) container.removeChild(container.firstChild);
      var svg = S("svg", { focusable: "false" });
      container.appendChild(svg);
      var h = draw(svg, w);
      svg.setAttribute("viewBox", "0 0 " + w + " " + h);
      svg.setAttribute("width", w);
      svg.setAttribute("height", h);
    }
    render();
    if ("ResizeObserver" in window) {
      var raf = 0;
      new ResizeObserver(function () {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(function () { render(false); });
      }).observe(container);
    }
    var api = { render: function () { render(true); } };
    charts.push(api);
    return api;
  }

  // ------------------------------------------------------------- tooltip

  var tip = document.getElementById("tooltip");
  var tipTimer = 0;
  function showTip(anchor, title, rows) {
    if (!tip) return;
    clearTimeout(tipTimer);
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    var h = document.createElement("div");
    h.className = "tooltip__title";
    h.textContent = title;
    tip.appendChild(h);
    (rows || []).forEach(function (r) {
      var row = document.createElement("div");
      row.className = "tooltip__row";
      var k = document.createElement("span");
      k.className = "tooltip__key";
      if (r.color) {
        var sw = document.createElement("span");
        sw.className = "tooltip__swatch";
        sw.style.background = "var(" + r.color + ")";
        k.appendChild(sw);
      }
      k.appendChild(document.createTextNode(r.label));
      var v = document.createElement("span");
      v.className = "tooltip__val";
      v.textContent = r.value;
      row.appendChild(k);
      row.appendChild(v);
      tip.appendChild(row);
    });
    tip.classList.add("is-on");
    placeTip(anchor);
  }
  function placeTip(anchor) {
    var x, y;
    if (anchor && anchor.clientX !== undefined) {
      x = anchor.clientX; y = anchor.clientY;
    } else if (anchor && anchor.getBoundingClientRect) {
      var r = anchor.getBoundingClientRect();
      x = r.left + r.width / 2; y = r.top;
    } else return;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var vw = document.documentElement.clientWidth, vh = window.innerHeight;
    var left = x + 14, top = y - th - 12;
    if (left + tw > vw - 8) left = x - tw - 14;
    if (left < 8) left = 8;
    if (top < 8) top = y + 18;
    if (top + th > vh - 8) top = vh - th - 8;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function hideTip() {
    clearTimeout(tipTimer);
    tipTimer = setTimeout(function () { if (tip) tip.classList.remove("is-on"); }, 60);
  }
  function bindTip(el, getContent) {
    el.setAttribute("data-tip", "");
    el.addEventListener("pointerenter", function (e) { var c = getContent(); showTip(e, c[0], c[1]); });
    el.addEventListener("pointermove", function (e) { if (e.pointerType !== "touch") placeTip(e); });
    el.addEventListener("pointerleave", function (e) { if (e.pointerType !== "touch") hideTip(); });
    el.addEventListener("click", function (e) { var c = getContent(); showTip(e, c[0], c[1]); });
    el.addEventListener("focus", function () { var c = getContent(); showTip(el, c[0], c[1]); });
    el.addEventListener("blur", hideTip);
  }
  window.addEventListener("scroll", function () { if (tip && tip.classList.contains("is-on")) tip.classList.remove("is-on"); }, { passive: true });
  document.addEventListener("pointerdown", function (e) {
    if (!e.target.closest || !e.target.closest("[data-tip], .heat-cell, .pairs button")) hideTip();
  });

  // --------------------------------------------------------- theme toggle

  (function theme() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var root = document.documentElement;
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    function effective() {
      var t = root.getAttribute("data-theme");
      return t === "dark" || t === "light" ? t : (mq.matches ? "dark" : "light");
    }
    var sun = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.8v2.1M12 19.1v2.1M4.9 4.9l1.5 1.5M17.6 17.6l1.5 1.5M2.8 12h2.1M19.1 12h2.1M4.9 19.1l1.5-1.5M17.6 6.4l1.5-1.5"/></svg>';
    var moon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.5 14.2A8.5 8.5 0 1 1 9.8 3.5a6.8 6.8 0 0 0 10.7 10.7z"/></svg>';
    function paint() {
      var dark = effective() === "dark";
      btn.innerHTML = dark ? sun : moon;
      btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    }
    btn.addEventListener("click", function () {
      var next = effective() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("gfb-theme", next); } catch (e) {}
      paint();
    });
    if (mq.addEventListener) mq.addEventListener("change", paint);
    paint();
  })();

  // ------------------------------------------------------ nav: state + spy

  (function nav() {
    var navEl = document.getElementById("top-nav");
    function onScroll() { if (navEl) navEl.classList.toggle("is-scrolled", window.scrollY > 8); }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    var links = $all("#nav-links a");
    if (!("IntersectionObserver" in window) || !links.length) return;
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { visible[en.target.id] = en.isIntersecting ? en.intersectionRatio : 0; });
      var best = null, bestTop = Infinity;
      Object.keys(map).forEach(function (id) {
        var sec = document.getElementById(id);
        if (!sec || !visible[id]) return;
        var top = Math.abs(sec.getBoundingClientRect().top);
        if (top < bestTop) { bestTop = top; best = id; }
      });
      links.forEach(function (a) { a.removeAttribute("aria-current"); });
      if (best && map[best]) map[best].setAttribute("aria-current", "true");
    }, { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.01] });
    Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  })();

  // --------------------------------------------------------------- tabs

  function setupTabs(tablist, onSelect) {
    if (!tablist) return;
    var tabs = $all('[role="tab"]', tablist);
    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
      });
      if (focus) tab.focus();
      onSelect(tab);
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(t, false); });
      t.addEventListener("keydown", function (e) {
        var j = null;
        if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = tabs.length - 1;
        if (j !== null) { e.preventDefault(); select(tabs[j], true); }
      });
    });
  }

  // Reference vs generated panels
  setupTabs($('#panels [role="tablist"]'), function (tab) {
    $all("#panels [role=tabpanel]").forEach(function (p) {
      p.hidden = p.id !== tab.getAttribute("aria-controls");
    });
  });

  // Code examples
  setupTabs(document.getElementById("code-tabs"), function (tab) {
    ["cp-load", "cp-dl", "cp-scores", "cp-eval", "cp-own"].forEach(function (id) {
      var p = document.getElementById(id);
      if (p) p.hidden = id !== tab.getAttribute("aria-controls");
    });
  });

  // Copy buttons
  $all(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var src;
      if (btn.dataset.copy) src = document.getElementById(btn.dataset.copy);
      else src = $all("pre.code").filter(function (p) { return p.id.indexOf("cp-") === 0 && !p.hidden; })[0];
      if (!src) return;
      var text = src.innerText.replace(/ /g, " ");
      var label = btn.querySelector("span");
      function done(msg) {
        if (label) label.textContent = msg;
        btn.classList.add("is-done");
        setTimeout(function () { if (label) label.textContent = "Copy"; btn.classList.remove("is-done"); }, 1600);
      }
      function fallback() {
        var r = document.createRange();
        r.selectNodeContents(src);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
        done("Selected");
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done("Copied"); }, fallback);
      } else fallback();
    });
  });

  // ------------------------------------------------------------ world map

  (function worldMap() {
    var svg = document.querySelector(".worldmap");
    if (!svg) return;
    $all(".city", svg).forEach(function (g) {
      var c = CITIES[+g.dataset.i];
      if (!c) return;
      bindTip(g, function () {
        var lat = Math.abs(c.lat).toFixed(2) + "° " + (c.lat >= 0 ? "N" : "S");
        var lon = Math.abs(c.lon).toFixed(2) + "° " + (c.lon >= 0 ? "E" : "W");
        return [c.name + ", " + c.country, [
          { label: "Named street blocks", value: String(c.blocks) },
          { label: "Reference assignments", value: c.refs.toLocaleString("en-US") },
          { label: "Drives on the", value: c.drive === "L" ? "left" : "right" },
          { label: "City center", value: lat + ", " + lon }
        ]];
      });
      g.addEventListener("pointerenter", function () { highlightCity(c.name, true); });
      g.addEventListener("pointerleave", function () { highlightCity(c.name, false); });
      g.addEventListener("focus", function () { highlightCity(c.name, true); });
      g.addEventListener("blur", function () { highlightCity(c.name, false); });
    });
    function highlightCity(name, on) {
      $all(".city", svg).forEach(function (g) {
        if (CITIES[+g.dataset.i].name === name) g.classList.toggle("is-active", on);
      });
    }
    if (!reduceMotion) {
      var start = function () { svg.classList.add("is-animated"); };
      if ("IntersectionObserver" in window) {
        var io = new IntersectionObserver(function (en) {
          if (en[0].isIntersecting) { start(); io.disconnect(); }
        }, { threshold: 0.25 });
        io.observe(svg);
      } else start();
    }
  })();

  // ------------------------------------------------------- coverage list

  (function coverage() {
    var list = document.getElementById("coverage-list");
    if (!list) return;
    var buttons = $all("#coverage .seg button");
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        buttons.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        var items = $all("li", list);
        items.sort(function (a, c) {
          if (b.dataset.sort === "name") return a.dataset.name.localeCompare(c.dataset.name);
          return (+c.dataset.refs) - (+a.dataset.refs) || a.dataset.name.localeCompare(c.dataset.name);
        });
        items.forEach(function (li) { list.appendChild(li); });
      });
    });
  })();

  // ------------------------------------------------ chart: hierarchy dots

  function drawHierarchy(svg, w) {
    var narrow = w < 520;
    var labelW = narrow ? 92 : 150;
    var padR = 26, top = 12, rowH = narrow ? 44 : 48, axisH = 34;
    var x0 = labelW, x1 = w - padR;
    var x = lin(0, 0.8, x0, x1);
    var h = top + rowH * HIERARCHY.length + axisH;
    var plotBottom = top + rowH * HIERARCHY.length;

    // target band
    S("rect", { x: 0, y: top, width: w, height: rowH, rx: 8, "class": "band-real", opacity: 0.55 }, svg);
    // grid
    [0, 0.2, 0.4, 0.6, 0.8].forEach(function (t) {
      S("line", { x1: x(t), x2: x(t), y1: top, y2: plotBottom, "class": t === 0 ? "ax-base" : "ax-grid" }, svg);
      T(svg, x(t), plotBottom + 20, t.toFixed(1), "ax-text", { "text-anchor": "middle" });
    });

    var rows = HIERARCHY.map(function (d, i) { return top + rowH * i + rowH / 2; });

    // drop connectors between the target row and the nearest same-city row
    var cy0 = rows[0], cy1 = rows[1];
    [["real", "s-real"], ["gen", "s-gen"]].forEach(function (s) {
      S("line", { x1: x(HIERARCHY[0][s[0]]), y1: cy0, x2: x(HIERARCHY[1][s[0]]), y2: cy1, "class": s[1], "stroke-width": 1.6, opacity: 0.55 }, svg);
    });

    HIERARCHY.forEach(function (d, i) {
      var cy = rows[i];
      T(svg, 4, cy + 5, narrow ? d.short : d.key, "row-text" + (d.target ? " row-text--strong" : ""));
      S("line", { x1: x(Math.min(d.real, d.gen)), x2: x(Math.max(d.real, d.gen)), y1: cy, y2: cy, "class": "ax-grid", "stroke-width": 3, "stroke-linecap": "round" }, svg);
      [["gen", "m-gen", "--gen", "Six-generator mean, L1"], ["real", "m-real", "--real", "Held-out real images"]].forEach(function (s) {
        var cx = x(d[s[0]]);
        S("circle", { cx: cx, cy: cy, r: 6.5, "class": s[1] + " m-ring" }, svg);
        var hit = S("circle", { cx: cx, cy: cy, r: 14, "class": "hit", tabindex: 0, role: "img", "aria-label": s[3] + ", " + d.key + ": " + f3(d[s[0]]) }, svg);
        bindTip(hit, function () {
          return [d.key, [
            { label: "Held-out real", value: f3(d.real), color: "--real" },
            { label: "Six-generator mean", value: f3(d.gen), color: "--gen" }
          ]];
        });
      });
      if (i === 0 || (i === 1 && x(d.real) - x(d.gen) > 52)) {
        var ly = i === 0 ? cy - 12 : cy + 22;
        T(svg, x(d.real), ly, f3(d.real), "val-text halo", { "text-anchor": "middle" });
        T(svg, x(d.gen), ly, f3(d.gen), "val-text halo", { "text-anchor": "middle" });
      }
    });

    // drop from the target row to the nearest same-city row, labelled beside each connector
    if (w >= 640) {
      var dReal = HIERARCHY[1].real - HIERARCHY[0].real;
      var dGen = HIERARCHY[1].gen - HIERARCHY[0].gen;
      var mxR = (x(HIERARCHY[0].real) + x(HIERARCHY[1].real)) / 2, myR = (cy0 + cy1) / 2;
      T(svg, mxR + 12, myR + 14, sgn(dReal) + " for real images", "note-text halo");
      T(svg, Math.min(x(HIERARCHY[0].gen), x(HIERARCHY[1].gen)) - 12, myR + 4, sgn(dGen) + " for generated", "note-text halo", { "text-anchor": "end" });
    }
    return h;
  }

  // ---------------------------------------------------- chart: slope L0-L2

  function drawSlope(svg, w) {
    var narrow = w < 520;
    var labelW = narrow ? 98 : 150;
    var top = 46, plotH = narrow ? 300 : 330, axisH = 30, padL = 42;
    var levels = ["L0", "L1", "L2"];
    var xs = [padL + 16, 0, 0];
    xs[2] = w - labelW - 14;
    xs[1] = (xs[0] + xs[2]) / 2;
    var y = lin(0.42, 0.55, top + plotH, top);
    var h = top + plotH + axisH;

    [0.42, 0.45, 0.48, 0.51, 0.54].forEach(function (t) {
      S("line", { x1: padL, x2: xs[2], y1: y(t), y2: y(t), "class": "ax-grid" }, svg);
      T(svg, padL - 8, y(t) + 4, t.toFixed(2), "ax-text", { "text-anchor": "end" });
    });
    levels.forEach(function (L, i) {
      S("line", { x1: xs[i], x2: xs[i], y1: top, y2: top + plotH, "class": "ax-base" }, svg);
      T(svg, xs[i], top + plotH + 21, L, "row-text row-text--strong", { "text-anchor": "middle" });
    });

    // delta annotations for the six-model mean
    var mean = levels.map(function (L) {
      return MODELS.reduce(function (a, m) { return a + m[L]; }, 0) / MODELS.length;
    });
    [["+0.036", "[0.029, 0.042]", 0], ["+0.004", "[0.001, 0.008]", 1]].forEach(function (a) {
      var mx = (xs[a[2]] + xs[a[2] + 1]) / 2;
      S("path", { d: "M" + (xs[a[2]] + 6) + " " + (top - 10) + "v-6H" + (xs[a[2] + 1] - 6) + "v6", fill: "none", "class": "ax-base" }, svg);
      T(svg, mx, top - 22, a[0] + (narrow ? "" : "  " + a[1]), "val-text", { "text-anchor": "middle" });
    });

    var lines = [];
    MODELS.forEach(function (m) {
      var d = levels.map(function (L, i) { return (i ? "L" : "M") + xs[i] + " " + y(m[L]); }).join(" ");
      var line = S("path", { d: d, "class": "line-model" }, svg);
      lines.push(line);
    });
    // mean line on top
    S("path", { d: mean.map(function (v, i) { return (i ? "L" : "M") + xs[i] + " " + y(v); }).join(" "), "class": "line-mean" }, svg);
    mean.forEach(function (v, i) { S("circle", { cx: xs[i], cy: y(v), r: 5, "class": "m-gen m-ring" }, svg); });

    // right labels with collision avoidance
    var items = MODELS.map(function (m, i) { return { i: i, y: y(m.L2), want: y(m.L2), text: m.name, v: m.L2 }; });
    items.sort(function (a, b) { return a.want - b.want; });
    var gap = 16;
    for (var k = 1; k < items.length; k++) if (items[k].y - items[k - 1].y < gap) items[k].y = items[k - 1].y + gap;
    for (k = items.length - 2; k >= 0; k--) if (items[k + 1].y - items[k].y < gap) items[k].y = items[k + 1].y - gap;
    var lx = xs[2] + 14;
    items.forEach(function (it) {
      var g = S("g", { "class": "lbl-model", tabindex: 0, role: "img", "aria-label": MODELS[it.i].name + ": L0 " + f3(MODELS[it.i].L0) + ", L1 " + f3(MODELS[it.i].L1) + ", L2 " + f3(MODELS[it.i].L2) }, svg);
      if (Math.abs(it.y - it.want) > 2) S("path", { d: "M" + (xs[2] + 3) + " " + it.want + "L" + (lx - 3) + " " + it.y, "class": "ax-base", fill: "none" }, g);
      T(g, lx, it.y + 4, it.text, "row-text");
      if (!narrow) T(g, w - 2, it.y + 4, f3(it.v), "val-text val-text--muted", { "text-anchor": "end" });
      S("rect", { x: lx - 4, y: it.y - 9, width: w - lx + 4, height: 18, "class": "hit" }, g);
      hover(g, it.i);
    });
    // wide invisible hit paths on lines
    MODELS.forEach(function (m, i) {
      var d = levels.map(function (L, j) { return (j ? "L" : "M") + xs[j] + " " + y(m[L]); }).join(" ");
      var hp = S("path", { d: d, fill: "none", stroke: "transparent", "stroke-width": 14, "pointer-events": "stroke" }, svg);
      hover(hp, i);
    });
    function hover(el, i) {
      var m = MODELS[i];
      var on = function () { svg.classList.add("is-focus"); lines.forEach(function (l, j) { l.classList.toggle("is-hot", j === i); l.classList.toggle("is-dim", j !== i); }); };
      var off = function () { svg.classList.remove("is-focus"); lines.forEach(function (l) { l.classList.remove("is-hot", "is-dim"); }); };
      el.addEventListener("pointerenter", on);
      el.addEventListener("pointerleave", off);
      el.addEventListener("focus", on);
      el.addEventListener("blur", off);
      bindTip(el, function () {
        return [m.name, [
          { label: "L0 city", value: f3(m.L0) },
          { label: "L1 + names", value: f3(m.L1) },
          { label: "L2 + GPS", value: f3(m.L2) },
          { label: "L1 − L0", value: sgn(m.L1 - m.L0) }
        ]];
      });
    }
    return h;
  }

  // ------------------------------------------------ chart: forest (generic)

  function forestPanel(svg, opt) {
    // opt: x0, x1, top, rows [{label, series: [{v, lo, hi, cls, hollow, stroke, color, name}]}],
    //      domain, ticks, ref, refLabel, title, rowH, labelX, showLabels,
    //      dec (tick decimals), vdec (value decimals), signed, valueLabels
    var x = lin(opt.domain[0], opt.domain[1], opt.x0, opt.x1);
    var y0 = opt.top + (opt.title ? 26 : 0);
    var rowH = opt.rowH;
    var bottom = y0 + rowH * opt.rows.length;
    if (opt.title) T(svg, opt.titleX === undefined ? opt.x0 : opt.titleX, opt.top + 13, opt.title, "row-text row-text--strong");
    opt.ticks.forEach(function (t) {
      S("line", { x1: x(t), x2: x(t), y1: y0, y2: bottom, "class": "ax-grid" }, svg);
      var lab = fmtTick(t, opt.dec);
      if (opt.refLabel && t === opt.ref) lab += " · " + opt.refLabel;
      T(svg, x(t), bottom + 18, lab, "ax-text", { "text-anchor": "middle" });
    });
    S("line", { x1: x(opt.ref), x2: x(opt.ref), y1: y0 - 4, y2: bottom, "class": "ax-ref" }, svg);
    opt.rows.forEach(function (r, i) {
      var cy = y0 + rowH * i + rowH / 2;
      if (opt.showLabels) T(svg, opt.labelX, cy + 5, r.label, "row-text");
      var n = r.series.length;
      r.series.forEach(function (s, j) {
        var yy = n > 1 ? cy + (j - (n - 1) / 2) * 14 : cy;
        S("line", { x1: x(s.lo), x2: x(s.hi), y1: yy, y2: yy, "class": s.stroke, "stroke-width": 2, "stroke-linecap": "round" }, svg);
        S("line", { x1: x(s.lo), x2: x(s.lo), y1: yy - 4, y2: yy + 4, "class": s.stroke, "stroke-width": 2 }, svg);
        S("line", { x1: x(s.hi), x2: x(s.hi), y1: yy - 4, y2: yy + 4, "class": s.stroke, "stroke-width": 2 }, svg);
        S("circle", { cx: x(s.v), cy: yy, r: s.hollow ? 5 : 5.5, "class": s.hollow ? s.hollow : (s.cls + " m-ring") }, svg);
        if (opt.valueLabels) T(svg, x(s.hi) + 8, yy + 4, sgn(s.v, opt.vdec), "val-text", {});
        var hit = S("rect", { x: x(s.lo) - 6, y: yy - 9, width: Math.max(18, x(s.hi) - x(s.lo) + 12), height: 18, "class": "hit", tabindex: 0, role: "img", "aria-label": r.label + ", " + s.name + ": " + s.v + " [" + s.lo + ", " + s.hi + "]" }, svg);
        bindTip(hit, function () {
          return [r.label, [
            { label: s.name, value: (opt.signed ? sgn(s.v, opt.vdec) : fmtTick(s.v, opt.vdec)), color: s.color },
            { label: "95% interval", value: "[" + fmtTick(s.lo, opt.vdec) + ", " + fmtTick(s.hi, opt.vdec) + "]" }
          ]];
        });
      });
    });
    return bottom + 26;
  }

  function drawControls(svg, w) {
    var wide = w >= 700;
    var labelW = wide ? 200 : Math.min(150, Math.max(118, w * 0.34));
    var rowH = 46, gap = 36;
    var panels = [
      { key: "cos", title: "CosSim benefit", domain: [-0.01, 0.02], ticks: w < 480 ? [-0.01, 0, 0.02] : [-0.01, 0, 0.01, 0.02] },
      { key: "ret", title: "Retrieval benefit", domain: [-0.01, 0.08], ticks: w < 480 ? [0, 0.04, 0.08] : [0, 0.02, 0.04, 0.06, 0.08] }
    ];
    function rows(key) {
      return CONTROLS.map(function (c) {
        return { label: w < 420 ? c.short : c.name, series: [{ v: c[key][0], lo: c[key][1], hi: c[key][2], cls: "m-gen", stroke: "s-gen", color: "--gen", name: "L1 minus control" }] };
      });
    }
    if (wide) {
      var pw = (w - labelW - gap) / 2;
      var h1 = forestPanel(svg, { x0: labelW, x1: labelW + pw - 44, top: 0, rows: rows("cos"), domain: panels[0].domain, ticks: panels[0].ticks, ref: 0, title: panels[0].title, rowH: rowH, labelX: 0, showLabels: true, dec: 2, vdec: 3, signed: true, valueLabels: true });
      var h2 = forestPanel(svg, { x0: labelW + pw + gap, x1: w - 48, top: 0, rows: rows("ret"), domain: panels[1].domain, ticks: panels[1].ticks, ref: 0, title: panels[1].title, rowH: rowH, labelX: 0, showLabels: false, dec: 2, vdec: 3, signed: true, valueLabels: true });
      return Math.max(h1, h2);
    }
    var ya = forestPanel(svg, { x0: labelW, x1: w - 52, top: 0, rows: rows("cos"), domain: panels[0].domain, ticks: panels[0].ticks, ref: 0, title: panels[0].title, rowH: rowH, labelX: 0, showLabels: true, dec: 2, vdec: 3, signed: true, valueLabels: true });
    var yb = forestPanel(svg, { x0: labelW, x1: w - 52, top: ya + 18, rows: rows("ret"), domain: panels[1].domain, ticks: panels[1].ticks, ref: 0, title: panels[1].title, rowH: rowH, labelX: 0, showLabels: true, dec: 2, vdec: 3, signed: true, valueLabels: true });
    return yb;
  }

  function drawClip(svg, w) {
    var narrow = w < 460;
    var labelW = narrow ? 112 : 150;
    var rowH = 50;
    function series(d) {
      return [
        { v: d.exact[0], lo: d.exact[1], hi: d.exact[2], cls: "m-clip", stroke: "s-clip", color: "--clip", name: "Exact prompt" },
        { v: d.loc[0], lo: d.loc[1], hi: d.loc[2], hollow: "m-hollow-clip", stroke: "s-clip", color: "--clip", name: "Location-only text" }
      ];
    }
    var ya = forestPanel(svg, {
      x0: labelW, x1: w - 12, top: 0, titleX: 0, title: "Spearman ρ with each outcome",
      rows: CLIP_RHO.map(function (d) { return { label: d.name, series: series(d) }; }),
      domain: [-0.15, 0.4], ticks: narrow ? [0, 0.2, 0.4] : [-0.1, 0, 0.1, 0.2, 0.3, 0.4], ref: 0, rowH: rowH, labelX: 0, showLabels: true, dec: 1, vdec: 3, signed: false
    });
    var yb = forestPanel(svg, {
      x0: labelW, x1: w - 12, top: ya + 16, titleX: 0, title: narrow ? "AUROC, all four retrieve the target" : "AUROC for “all four images retrieve the target”",
      rows: [{ label: CLIP_AUC.name, series: series(CLIP_AUC) }],
      domain: [0.4, 0.7], ticks: [0.4, 0.5, 0.6, 0.7], ref: 0.5, refLabel: "chance", rowH: rowH, labelX: 0, showLabels: true, dec: 1, vdec: 3, signed: false
    });
    return yb;
  }

  function drawQuartile(svg, w) {
    var labelW = w < 420 ? 128 : 158;
    var rowH = 38, top = 4;
    var x = lin(0, 60, labelW, w - 52);
    var bottom = top + rowH * QUARTILE.length;
    [0, 20, 40, 60].forEach(function (t) {
      S("line", { x1: x(t), x2: x(t), y1: top, y2: bottom, "class": t === 0 ? "ax-base" : "ax-grid" }, svg);
      T(svg, x(t), bottom + 18, t + "%", "ax-text", { "text-anchor": "middle" });
    });
    QUARTILE.forEach(function (q, i) {
      var cy = top + rowH * i + rowH / 2;
      T(svg, 0, cy + 5, q.name, "row-text" + (i === 1 ? " row-text--strong" : ""));
      var bw = x(q.v) - x(0);
      S("path", { d: "M" + x(0) + " " + (cy - 8) + "h" + (bw - 4) + "a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4h" + (-(bw - 4)) + "z", "class": "m-clip", opacity: i === 1 ? 1 : 0.72 }, svg);
      T(svg, x(q.v) + 7, cy + 5, q.v.toFixed(1) + "%", "val-text");
      var hit = S("rect", { x: 0, y: cy - rowH / 2, width: w, height: rowH, "class": "hit", tabindex: 0, role: "img", "aria-label": q.name + ": " + q.v + " percent of panels have no image that retrieves the target" }, svg);
      bindTip(hit, function () { return [q.name, [{ label: "No image retrieves target", value: q.v.toFixed(1) + "%", color: "--clip" }]]; });
    });
    return bottom + 26;
  }

  // ----------------------------------------------------- chart: scatter

  function drawScatter(svg, w) {
    var narrow = w < 560;
    var padL = 50, padR = narrow ? 14 : 24, top = 14, plotH = narrow ? 300 : 360, axisH = 46;
    var x = lin(0.3, 0.95, padL, w - padR);
    var y = lin(0.1, 0.9, top + plotH, top);
    var h = top + plotH + axisH;
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].forEach(function (t) {
      S("line", { x1: x(t), x2: x(t), y1: top, y2: top + plotH, "class": "ax-grid" }, svg);
      if (!narrow || [0.3, 0.5, 0.7, 0.9].indexOf(t) >= 0) T(svg, x(t), top + plotH + 18, t.toFixed(1), "ax-text", { "text-anchor": "middle" });
    });
    [0.1, 0.3, 0.5, 0.7, 0.9].forEach(function (t) {
      S("line", { x1: padL, x2: w - padR, y1: y(t), y2: y(t), "class": "ax-grid" }, svg);
      T(svg, padL - 8, y(t) + 4, t.toFixed(1), "ax-text", { "text-anchor": "end" });
    });
    S("line", { x1: padL, x2: w - padR, y1: top + plotH, y2: top + plotH, "class": "ax-base" }, svg);
    S("line", { x1: padL, x2: padL, y1: top, y2: top + plotH, "class": "ax-base" }, svg);
    T(svg, (padL + w - padR) / 2, top + plotH + 38, "CosSim to the target panel", "ax-title", { "text-anchor": "middle" });
    T(svg, 12, top + plotH / 2, "Top-1 retrieval accuracy", "ax-title", { "text-anchor": "middle", transform: "rotate(-90 12 " + (top + plotH / 2) + ")" });

    // generator cluster hull (ellipse)
    var gx = BOARD.gens.map(function (g) { return x(g.cos); }), gy = BOARD.gens.map(function (g) { return y(g.ret); });
    var cx = (Math.min.apply(null, gx) + Math.max.apply(null, gx)) / 2, cy = (Math.min.apply(null, gy) + Math.max.apply(null, gy)) / 2;
    var rx = (Math.max.apply(null, gx) - Math.min.apply(null, gx)) / 2 + 16, ry = (Math.max.apply(null, gy) - Math.min.apply(null, gy)) / 2 + 14;
    S("ellipse", { cx: cx, cy: cy, rx: rx, ry: ry, "class": "band-gen" }, svg);
    T(svg, cx, cy - ry - 8, "Six generators", "note-text halo", { "text-anchor": "middle" });

    BOARD.gens.forEach(function (g) {
      S("circle", { cx: x(g.cos), cy: y(g.ret), r: 5, "class": "m-gen m-ring" }, svg);
    });
    BOARD.gens.forEach(function (g) {
      var hit = S("circle", { cx: x(g.cos), cy: y(g.ret), r: 9, "class": "hit", tabindex: 0, role: "img", "aria-label": g.name + ": CosSim " + f3(g.cos) + ", retrieval " + f3(g.ret) }, svg);
      bindTip(hit, function () { return [g.name + ", L1", [{ label: "CosSim", value: f3(g.cos), color: "--gen" }, { label: "Retrieval", value: f3(g.ret), color: "--gen" }]]; });
    });

    var anchorLabels = {
      "Held-out Real": { dx: -12, dy: 4, anchor: "end", text: "Held-out real, same block" },
      "Random-Same-Country": narrow ? { dx: -12, dy: 4, anchor: "end", text: "Random real, same country" } : { dx: 0, dy: 24, anchor: "middle", text: "Random real, same country" },
      "Random-Global": { dx: 12, dy: 4, anchor: "start", text: "Random real, global" }
    };
    BOARD.anchors.forEach(function (a) {
      var px = x(a.cos), py = y(a.ret);
      S("circle", { cx: px, cy: py, r: a.same ? 7 : 6, "class": a.same ? "m-real m-ring" : "m-hollow-real" }, svg);
      var L = anchorLabels[a.name];
      T(svg, px + L.dx, py + L.dy, L.text, "note-text halo", { "text-anchor": L.anchor });
      var hit = S("circle", { cx: px, cy: py, r: 13, "class": "hit", tabindex: 0, role: "img", "aria-label": a.name + ": CosSim " + f3(a.cos) + ", retrieval " + f3(a.ret) }, svg);
      bindTip(hit, function () { return [a.name, [{ label: "CosSim", value: f3(a.cos), color: "--real" }, { label: "Retrieval", value: f3(a.ret), color: "--real" }]]; });
    });
    return h;
  }

  // ----------------------------------------------------- chart: heatmap

  var heatOrder = "alpha";
  var heatHot = null;
  function drawHeat(svg, w) {
    if (!CROSS) return 10;
    var n = CITIES.length;
    var narrow = w < 560;
    var labelW = narrow ? 34 : 98;
    var topH = narrow ? 30 : 92;
    var cell = Math.floor(Math.min((w - labelW - 4) / n, 26));
    var size = cell * n;
    var x0 = labelW, y0 = topH;
    var order = heatOrder === "cluster" && CROSS_CLUSTER ? CROSS_CLUSTER.slice() : CITIES.map(function (c, i) { return i; });
    var rowLabels = [], colLabels = [];
    order.forEach(function (ci, k) {
      var c = CITIES[ci];
      var lbl = narrow ? c.code : c.name;
      rowLabels.push(T(svg, x0 - 6, y0 + k * cell + cell / 2 + 4, lbl, "heat-label", { "text-anchor": "end", style: narrow ? "font-size:9.5px" : null }));
      var cx = x0 + k * cell + cell / 2;
      if (narrow) colLabels.push(null);
      else colLabels.push(T(svg, cx, y0 - 8, lbl, "heat-label", { transform: "rotate(-60 " + cx + " " + (y0 - 8) + ")" }));
    });
    var cells = [];
    order.forEach(function (ri, r) {
      order.forEach(function (cj, c) {
        var v = CROSS[ri][cj];
        var diag = ri === cj;
        var t = Math.max(0, Math.min(1, v / 0.8));
        var rect = S("rect", {
          x: x0 + c * cell, y: y0 + r * cell, width: cell, height: cell,
          "class": "heat-cell" + (diag ? " is-diag" : ""),
          style: diag ? null : "fill: color-mix(in oklab, var(--heat-1) " + Math.round(t * 100) + "%, var(--heat-0))"
        }, svg);
        rect.dataset.r = r; rect.dataset.c = c;
        cells.push(rect);
        if (diag) S("line", { x1: x0 + c * cell + 3, y1: y0 + r * cell + cell - 3, x2: x0 + c * cell + cell - 3, y2: y0 + r * cell + 3, "class": "heat-diag" }, svg);
        if (!diag) {
          var show = function (e) { hot(r, c); showTip(e, CITIES[ri].name + " × " + CITIES[cj].name, [{ label: "Cosine similarity", value: v.toFixed(2) }]); };
          rect.addEventListener("pointerenter", show);
          rect.addEventListener("click", show);
          rect.addEventListener("pointermove", function (e) { if (e.pointerType !== "touch") placeTip(e); });
          rect.addEventListener("pointerleave", function (e) { if (e.pointerType !== "touch") { cold(); hideTip(); } });
        }
      });
    });
    function hot(r, c) {
      cold();
      var rect = cells[r * n + c];
      rect.classList.add("is-hot");
      rect.parentNode.appendChild(rect);
      rowLabels[r].classList.add("is-hot");
      if (colLabels[c]) colLabels[c].classList.add("is-hot");
      heatHot = [r, c];
    }
    function cold() {
      cells.forEach(function (x) { x.classList.remove("is-hot"); });
      rowLabels.forEach(function (x) { x.classList.remove("is-hot"); });
      colLabels.forEach(function (x) { if (x) x.classList.remove("is-hot"); });
      heatHot = null;
    }
    svg.__hotPair = function (i, j) {
      var r = order.indexOf(i), c = order.indexOf(j);
      hot(r, c);
      var rect = cells[r * n + c];
      showTip(rect, CITIES[i].name + " × " + CITIES[j].name, [{ label: "Cosine similarity", value: CROSS[i][j].toFixed(2) }]);
    };
    svg.__cold = function () { cold(); hideTip(); };
    return y0 + size + 6;
  }

  // ----------------------------------------------------------- leaderboard

  var boardSort = { key: "cos", dir: "desc" };
  function renderBoard() {
    var body = document.getElementById("board-body");
    if (!body) return;
    var all = BOARD.gens.concat([BOARD.mean], BOARD.anchors);
    var max = {}, best = {};
    BOARD_COLS.forEach(function (c) {
      max[c.key] = Math.max.apply(null, all.map(function (r) { return r[c.key]; }));
      var vals = BOARD.gens.map(function (g) { return g[c.key]; });
      best[c.key] = c.higher ? Math.max.apply(null, vals) : Math.min.apply(null, vals);
    });
    var gens = BOARD.gens.slice().sort(function (a, b) {
      return boardSort.dir === "desc" ? b[boardSort.key] - a[boardSort.key] : a[boardSort.key] - b[boardSort.key];
    });
    while (body.firstChild) body.removeChild(body.firstChild);
    function groupRow(text) {
      var tr = document.createElement("tr");
      tr.className = "group-row";
      var td = document.createElement("td");
      td.colSpan = 6;
      td.textContent = text;
      tr.appendChild(td);
      body.appendChild(tr);
    }
    function row(r, cls, rank) {
      var tr = document.createElement("tr");
      if (cls) tr.className = cls;
      var th = document.createElement("th");
      th.scope = "row";
      th.className = "model";
      if (rank) {
        var rk = document.createElement("span");
        rk.className = "rank";
        rk.textContent = rank;
        th.appendChild(rk);
      }
      th.appendChild(document.createTextNode(r.name));
      tr.appendChild(th);
      BOARD_COLS.forEach(function (c) {
        var td = document.createElement("td");
        var wrap = document.createElement("div");
        wrap.className = "cellbar";
        if (!cls && r[c.key] === best[c.key]) {
          wrap.className += " is-best";
          wrap.title = "Best generator on this metric";
        }
        var b = document.createElement("b");
        b.textContent = f3(r[c.key]);
        var bar = document.createElement("i");
        bar.style.width = (100 * r[c.key] / max[c.key]).toFixed(1) + "%";
        bar.setAttribute("aria-hidden", "true");
        var track = document.createElement("span");
        track.appendChild(bar);
        wrap.appendChild(b);
        wrap.appendChild(track);
        td.appendChild(wrap);
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
    groupRow("Generators · L1 named-block prompts");
    gens.forEach(function (g, i) { row(g, "", String(i + 1)); });
    row(BOARD.mean, "is-mean");
    groupRow("Real-image anchors");
    BOARD.anchors.forEach(function (a) { row(a, "is-real"); });
  }
  (function board() {
    var table = document.getElementById("board-table");
    if (!table) return;
    var ths = $all("thead th[data-key]", table);
    ths.forEach(function (th) {
      th.querySelector("button").addEventListener("click", function () {
        var key = th.dataset.key;
        if (boardSort.key === key) boardSort.dir = boardSort.dir === "desc" ? "asc" : "desc";
        else { boardSort.key = key; boardSort.dir = th.dataset.dir; }
        ths.forEach(function (o) {
          var on = o === th;
          o.setAttribute("aria-sort", on ? (boardSort.dir === "desc" ? "descending" : "ascending") : "none");
          o.querySelector(".sort-ind").textContent = on ? (boardSort.dir === "desc" ? "▾" : "▴") : "";
        });
        renderBoard();
      });
    });
    renderBoard();
  })();

  // ------------------------------------------------------ qualitative grid

  (function qualitative() {
    var tablist = document.getElementById("qual-tabs");
    if (!tablist) return;
    var real = document.getElementById("qual-real");
    var cityName = document.getElementById("qual-city-name");
    var gens = document.getElementById("qual-gens");
    var panel = document.getElementById("qp");
    var grid = document.getElementById("qual-grid");
    var btnOne = document.getElementById("qual-one"), btnAll = document.getElementById("qual-all");
    setupTabs(tablist, function (tab) {
      var city = tab.dataset.city;
      panel.setAttribute("aria-labelledby", tab.id);
      real.src = "static/images/qualitative/" + city + "_real.jpg";
      real.alt = "Real Mapillary reference image, " + QUAL_CITIES[city];
      cityName.textContent = QUAL_CITIES[city];
      $all("img", gens).forEach(function (img) {
        var m = img.dataset.model;
        img.src = "static/images/qualitative/" + city + "_" + m + ".jpg";
        img.alt = img.dataset.name + " image for a city-only prompt, " + QUAL_CITIES[city];
      });
      if (!grid.hidden) setLayout(false);
    });
    function setLayout(all) {
      grid.hidden = !all;
      panel.hidden = all;
      btnAll.setAttribute("aria-pressed", all ? "true" : "false");
      btnOne.setAttribute("aria-pressed", all ? "false" : "true");
    }
    btnOne.addEventListener("click", function () { setLayout(false); });
    btnAll.addEventListener("click", function () { setLayout(true); });
  })();

  // ------------------------------------------------------------ mount all

  var registry = {
    hierarchy: drawHierarchy,
    slope: drawSlope,
    controls: drawControls,
    clip: drawClip,
    quartile: drawQuartile,
    scatter: drawScatter,
    heat: drawHeat
  };
  var heatChart = null;
  $all("[data-chart]").forEach(function (el) {
    var fn = registry[el.dataset.chart];
    if (!fn) return;
    var api = mount(el, fn);
    if (el.dataset.chart === "heat") heatChart = { api: api, el: el };
  });

  // heatmap order toggle and pair lists
  (function heatControls() {
    if (!CROSS) return;
    var btns = $all("#chart-heat .seg button");
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        heatOrder = b.dataset.order;
        btns.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        if (heatChart) heatChart.api.render();
      });
    });
    var pairs = [];
    for (var i = 0; i < CITIES.length; i++) for (var j = i + 1; j < CITIES.length; j++) pairs.push([CROSS[i][j], i, j]);
    pairs.sort(function (a, b) { return b[0] - a[0]; });
    function fill(listId, arr) {
      var ol = document.getElementById(listId);
      if (!ol) return;
      arr.forEach(function (p) {
        var li = document.createElement("li");
        var b = document.createElement("button");
        b.type = "button";
        var a = document.createElement("span");
        a.textContent = CITIES[p[1]].name + " · " + CITIES[p[2]].name;
        var v = document.createElement("span");
        v.textContent = (p[0] < 0 ? "−" : "") + Math.abs(p[0]).toFixed(2);
        b.appendChild(a);
        b.appendChild(v);
        var on = function () {
          $all(".pairs button").forEach(function (x) { x.classList.toggle("is-active", x === b); });
          var svg = heatChart && heatChart.el.querySelector("svg");
          if (svg && svg.__hotPair) svg.__hotPair(p[1], p[2]);
        };
        var off = function () {
          b.classList.remove("is-active");
          var svg = heatChart && heatChart.el.querySelector("svg");
          if (svg && svg.__cold) svg.__cold();
        };
        b.addEventListener("pointerenter", on);
        b.addEventListener("focus", on);
        b.addEventListener("click", on);
        b.addEventListener("pointerleave", off);
        b.addEventListener("blur", off);
        li.appendChild(b);
        ol.appendChild(li);
      });
    }
    fill("pairs-top", pairs.slice(0, 5));
    fill("pairs-bottom", pairs.slice(-5).reverse());
  })();

  // re-render once web fonts are ready so measured text is accurate
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () { charts.forEach(function (c) { c.render(); }); });
  }
})();
