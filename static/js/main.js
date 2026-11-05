/* MetaboNet CGM forecasting — page behaviour. No dependencies.
   Data: window.MBN_RESULTS (static/data/results.js), window.SITE (site-config.js). */
(function () {
  "use strict";
  const R = window.MBN_RESULTS || { live: {}, loso: {}, atlas: [] };
  const S = window.SITE || {};
  const H = ["30", "60", "90", "120"];
  const $ = (q, el) => (el || document).querySelector(q);
  const $$ = (q, el) => Array.from((el || document).querySelectorAll(q));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage blocked */ } },
  };

  /* ------------------------------------------------------- descriptions -- */
  const DESC = {
    persistence: "Predict the current CGM value at every horizon (reference, no parameters)",
    lightgbm_whatif: "LightGBM on 240 features incl. known-future insulin & carbs; L1 on log-ratio",
    lstm_s0: "LSTM over 24 h history + known-future dose block; seed 0",
    lstm_s1: "Same LSTM, seed 1",
    lstm_s2: "Same LSTM, seed 2",
    seedavg: "Mean of the three LSTM seeds",
    "seedavg+gbm": "0.5 × seed-averaged LSTM + 0.5 × LightGBM (= annual_blendseed recipe)",
    blend50: "0.5 × single-seed LSTM + 0.5 × LightGBM",
    dual_s0: "Screen: two-stream encoder (history and dose streams encoded separately)",
    tcn_s0: "Screen: temporal convolutional encoder in place of the LSTM",
    transformer_s0: "Screen: transformer encoder in place of the LSTM",
    lstm_e50_s0: "Screen: 50-epoch cosine schedule (confounds length with LR schedule)",
    lstm_dz25_s0: "Screen: checkpoint chosen on stop-split DTS-A, 25 epochs",
    lstm_dz50_s0: "Screen: checkpoint chosen on stop-split DTS-A, 50 epochs",
    lstm_futseq_s0: "Screen: GRU decoder reads the known-future schedule in order",
    live_causal: "LightGBM without any future inputs (past only)",
    live_whatif: "LightGBM what-if — one booster per horizon × 3 seeds, 240 features",
    live_lstm: "LSTM, seed 0",
    live_lstm_s1: "LSTM, seed 1",
    live_lstm_s2: "LSTM, seed 2",
    live_lstm_s3: "LSTM, seed 3",
    live_lstm_s4: "LSTM, seed 4",
    live_lstm_s5: "LSTM, seed 5",
    live_lstmseed6: "Mean of six LSTM seeds — best entry",
    live_blend50: "0.5 × LSTM seed 0 + 0.5 × LightGBM what-if",
    live_blendseed: "0.5 × 3-seed LSTM + 0.5 × LightGBM what-if",
    live_blendseed6: "0.5 × 6-seed LSTM + 0.5 × LightGBM what-if",
  };
  const MLAB = {
    dtsa: "DTS-A (%)", mard: "MARD (%)", rmse: "RMSE (mg/dL)", mse: "MSE (mg/dL)²",
  };
  const HIGHER = { dtsa: true, mard: false, rmse: false, mse: false };
  const fmt = (m, v) => (v == null || isNaN(v) ? "—" : m === "mse" ? v.toFixed(1) : v.toFixed(2));
  const fmtInt = (n) => n.toLocaleString("en-US");

  /* ------------------------------------------------------ site config ---- */
  function applyConfig() {
    const who = S.anonymous ? S.anon : S.named;
    if (!who) return;
    const a = $("#authors");
    if (a) {
      a.innerHTML = who.authors
        .map((p) => `${p.name}<sup>${p.aff.join(",")}</sup>`)
        .join(", ");
    }
    const af = $("#affils");
    if (af) af.innerHTML = who.affiliations.map((x, i) => `<sup>${i + 1}</sup>${x}`).join(" &nbsp; ");
    $$("[data-team]").forEach((el) => (el.textContent = who.team));
    const ack = $("#ack");
    if (ack) ack.textContent = who.acknowledgements;
    const L = S.links || {};
    $$("[data-link]").forEach((el) => {
      const url = L[el.dataset.link];
      if (!url || url === "#") {
        el.classList.add("disabled");
        el.removeAttribute("href");
        el.title = "Coming soon";
      } else {
        el.href = url;
      }
    });
    const b = S.bib || {};
    const bib = $("#bibtex");
    if (bib) {
      bib.textContent =
        `@article{${b.key},\n  title   = {${b.title}},\n  author  = {${who.bibAuthor}},\n` +
        `  journal = {${b.venue}},\n  year    = {${b.year}},\n  url     = {${b.url}}\n}`;
    }
  }

  /* ------------------------------------------------------------ theme ---- */
  function initTheme() {
    const saved = store.get("mbn-theme");
    if (saved) document.documentElement.dataset.theme = saved;
    const btn = $("#themeBtn");
    if (!btn) return;
    const label = () => {
      const t = document.documentElement.dataset.theme ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      btn.textContent = t === "dark" ? "☀︎" : "☾";
      btn.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
    };
    label();
    btn.addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      store.set("mbn-theme", next);
      label();
    });
  }

  /* --------------------------------------------------- nav + scrollspy --- */
  function initNav() {
    const nav = $("#nav");
    const mb = $("#menuBtn");
    if (mb) mb.addEventListener("click", () => nav.classList.toggle("open"));
    $$("#nav a").forEach((a) => a.addEventListener("click", () => nav.classList.remove("open")));
    const links = $$("#nav a[href^='#']");
    const secs = links.map((a) => $(a.getAttribute("href"))).filter(Boolean);
    const onScroll = () => {
      const y = window.scrollY + 90;
      let cur = null;
      secs.forEach((s) => { if (s.offsetTop <= y) cur = s.id; });
      links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + cur));
      const tt = $("#toTop");
      if (tt) tt.classList.toggle("show", window.scrollY > 800);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    const tt = $("#toTop");
    if (tt) tt.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  }

  /* -------------------------------------------------------- count-up ----- */
  function initCounters() {
    const els = $$("[data-count]");
    const run = (el) => {
      const target = parseFloat(el.dataset.count);
      const dec = parseInt(el.dataset.dec || "0", 10);
      const pre = el.dataset.pre || "";
      const suf = el.dataset.suf || "";
      const t0 = performance.now();
      const dur = 1100;
      const step = (t) => {
        const k = Math.min(1, (t - t0) / dur);
        const e = 1 - Math.pow(1 - k, 3);
        el.textContent = pre + (target * e).toFixed(dec) + suf;
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    if (!("IntersectionObserver" in window)) { els.forEach(run); return; }
    const io = new IntersectionObserver((ents) => {
      ents.forEach((en) => { if (en.isIntersecting) { run(en.target); io.unobserve(en.target); } });
    }, { threshold: 0.4 });
    els.forEach((el) => {
      // render the final value first so the page reads correctly without JS animation
      const dec = parseInt(el.dataset.dec || "0", 10);
      el.textContent = (el.dataset.pre || "") + parseFloat(el.dataset.count).toFixed(dec) + (el.dataset.suf || "");
      io.observe(el);
    });
  }

  /* ------------------------------------------------------- hero bars ----- */
  function initHeroBars() {
    const box = $("#heroBars");
    if (!box || !R.live.live_lstmseed6) return;
    const best = R.live.live_lstmseed6, pers = R.live.persistence;
    const k = 0.86; // leave room for the value label
    box.innerHTML = H.map((h) => {
      const p = pers[h].dtsa, m = best[h].dtsa;
      return `<div class="hbar"><div class="lab">${h} min</div><div class="track">
        <div class="b p" style="width:0" data-w="${p * k}"></div>
        <div class="v" style="left:calc(${p * k}% + 6px);top:-2px;color:#cbd5e1">${p.toFixed(1)}</div>
        <div class="b m" style="width:0" data-w="${m * k}"></div>
        <div class="v" style="left:calc(${m * k}% + 6px);top:13px;font-weight:700">${m.toFixed(1)}</div>
      </div></div>`;
    }).join("");
    setTimeout(() => $$(".b", box).forEach((b) => (b.style.width = b.dataset.w + "%")), 120);
  }

  /* ----------------------------------------------------- table helpers --- */
  function makeSortable(table) {
    const ths = $$("th.sortable", table);
    ths.forEach((th, idx) => {
      th.addEventListener("click", () => {
        const col = parseInt(th.dataset.col, 10);
        const asc = !th.classList.contains("asc");
        ths.forEach((t) => t.classList.remove("asc", "desc"));
        th.classList.add(asc ? "asc" : "desc");
        const body = table.tBodies[0];
        const rows = Array.from(body.rows);
        rows.sort((a, b) => {
          const x = a.cells[col].dataset.v, y = b.cells[col].dataset.v;
          const nx = parseFloat(x), ny = parseFloat(y);
          const c = !isNaN(nx) && !isNaN(ny) ? nx - ny : String(x).localeCompare(String(y));
          return asc ? c : -c;
        });
        rows.forEach((r) => body.appendChild(r));
      });
    });
  }

  function metricTable(models, block, metric, opts) {
    opts = opts || {};
    const cols = H.concat(["mean"]);
    const best = {};
    cols.forEach((c) => {
      const vals = models.filter((m) => !opts.skipBest || !opts.skipBest.includes(m))
        .map((m) => block[m] && block[m][c] ? block[m][c][metric] : null).filter((v) => v != null);
      best[c] = HIGHER[metric] ? Math.max(...vals) : Math.min(...vals);
    });
    const head = `<thead><tr><th class="sortable" data-col="0">Model</th>${cols
      .map((c, i) => `<th class="num sortable" data-col="${i + 1}">${c === "mean" ? "Mean" : c + " min"}</th>`)
      .join("")}</tr></thead>`;
    const rows = models.map((m) => {
      const rec = block[m];
      const cls = [opts.highlight && opts.highlight.includes(m) ? "hl" : "",
                   opts.ref && opts.ref.includes(m) ? "ref" : ""].join(" ");
      return `<tr class="${cls}"><td data-v="${m}"><span class="mname">${m}</span>
          <span class="mdesc">${DESC[m] || ""}</span></td>${cols.map((c) => {
            const v = rec && rec[c] ? rec[c][metric] : null;
            const isBest = v != null && Math.abs(v - best[c]) < 1e-9 && !(opts.ref || []).includes(m);
            return `<td class="num${isBest ? " best" : ""}" data-v="${v == null ? "" : v}">${fmt(metric, v)}</td>`;
          }).join("")}</tr>`;
    }).join("");
    return `<table>${head}<tbody>${rows}</tbody></table>`;
  }

  function seg(container, onChange) {
    $$("button", container).forEach((b) => b.addEventListener("click", () => {
      $$("button", container).forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      onChange(b.dataset.v);
    }));
  }

  /* ------------------------------------------------------ live table ---- */
  const LIVE_ORDER = ["live_lstmseed6", "live_blendseed6", "live_blendseed", "live_blend50",
    "live_lstm_s3", "live_lstm_s5", "live_lstm_s4", "live_lstm", "live_lstm_s1", "live_lstm_s2",
    "live_whatif", "live_causal", "persistence"];

  function initLive() {
    const box = $("#liveTable");
    if (!box) return;
    let metric = "dtsa";
    const draw = () => {
      const models = LIVE_ORDER.filter((m) => R.live[m]);
      box.innerHTML = metricTable(models, R.live, metric,
        { highlight: ["live_lstmseed6"], ref: ["persistence"], skipBest: ["persistence"] });
      makeSortable($("table", box));
      $("#liveMetricLabel").textContent = MLAB[metric];
    };
    seg($("#liveSeg"), (v) => { metric = v; draw(); });
    draw();
  }

  /* ---------------------------------------------------------- arena ------ */
  function initArena() {
    const a = $("#arenaA"), b = $("#arenaB");
    if (!a) return;
    const models = LIVE_ORDER.filter((m) => R.live[m]);
    const opt = models.map((m) => `<option value="${m}">${m}</option>`).join("");
    a.innerHTML = opt; b.innerHTML = opt;
    a.value = "live_lstmseed6"; b.value = "live_whatif";
    let metric = "dtsa";
    const draw = () => {
      const A = R.live[a.value], B = R.live[b.value];
      const all = H.map((h) => [A[h][metric], B[h][metric]]).flat();
      const lo = metric === "dtsa" ? Math.min(40, ...all) : 0;
      const hi = Math.max(...all) * (metric === "dtsa" ? 1 : 1.08);
      const w = (v) => Math.max(2, ((v - lo) / (hi - lo)) * 100);
      const side = (M, col) => H.map((h) => `<div class="abar"><span>${h} min</span>
          <div class="t"><div class="f" style="width:${w(M[h][metric])}%;background:${col}"></div></div>
          <span class="v">${fmt(metric, M[h][metric])}</span></div>`).join("");
      $("#arenaPanelA").innerHTML = `<h4 class="mname">${a.value}</h4><p class="small muted">${DESC[a.value] || ""}</p>${side(A, "var(--accent)")}`;
      $("#arenaPanelB").innerHTML = `<h4 class="mname">${b.value}</h4><p class="small muted">${DESC[b.value] || ""}</p>${side(B, "var(--amber)")}`;
      const dl = H.concat(["mean"]).map((h) => {
        const d = A[h][metric] - B[h][metric];
        const good = HIGHER[metric] ? d > 0 : d < 0;
        const cls = Math.abs(d) < 1e-9 ? "" : good ? "up" : "down";
        return `<td class="num"><span class="delta ${cls}">${d > 0 ? "+" : ""}${fmt(metric, d)}</span></td>`;
      }).join("");
      $("#arenaDelta").innerHTML = `<table><thead><tr><th>A − B, ${MLAB[metric]}</th>${H.map((h) => `<th class="num">${h} min</th>`).join("")}<th class="num">Mean</th></tr></thead>
        <tbody><tr><td><span class="mname">${a.value}</span> − <span class="mname">${b.value}</span></td>${dl}</tr></tbody></table>`;
    };
    a.addEventListener("change", draw); b.addEventListener("change", draw);
    seg($("#arenaSeg"), (v) => { metric = v; draw(); });
    draw();
  }

  /* ----------------------------------------------------------- LOSO ------ */
  const LOSO_ORDER = ["seedavg+gbm", "blend50", "seedavg", "lstm_s0", "lstm_s1", "lstm_s2",
    "lstm_dz25_s0", "lstm_dz50_s0", "lstm_e50_s0", "lstm_futseq_s0", "dual_s0", "tcn_s0",
    "transformer_s0", "lightgbm_whatif", "persistence"];
  const FOLD_INFO = {
    CTR3: "26 subjects · 62,846 anchors · held out of training: 0.9% of FIT rows",
    IOBP2: "294 subjects · 432,062 anchors · iLet Bionic Pancreas — the annual cohort's device",
    Loop: "749 subjects · 5,180,479 anchors · holding it out removes 73% of training data",
    PEDAP: "59 subjects · 230,944 anchors · paediatric closed-loop cohort",
    ReplaceBG: "184 subjects · 1,059,592 anchors · manual dosing, unannounced meals",
  };

  function initLoso() {
    const box = $("#losoTable");
    if (!box) return;
    let fold = "IOBP2", metric = "dtsa";
    const sel = $("#losoFold");
    sel.innerHTML = Object.keys(R.loso).map((f) => `<option>${f}</option>`).join("");
    sel.value = fold;
    const draw = () => {
      const blk = R.loso[fold];
      const models = LOSO_ORDER.filter((m) => blk[m]);
      box.innerHTML = metricTable(models, blk, metric,
        { highlight: ["seedavg+gbm", "seedavg"], ref: ["persistence"], skipBest: ["persistence"] });
      makeSortable($("table", box));
      $("#losoInfo").textContent = FOLD_INFO[fold] || "";
      $("#losoMetricLabel").textContent = MLAB[metric];
    };
    sel.addEventListener("change", () => { fold = sel.value; draw(); });
    seg($("#losoSeg"), (v) => { metric = v; draw(); });
    draw();
  }

  /* ----------------------------------------------------------- atlas ----- */
  function initAtlas() {
    const kindSel = $("#atKind"), splitSel = $("#atSplit"), levelSel = $("#atLevel");
    const frame = $("#atFrame"), open = $("#atOpen");
    if (!kindSel || !R.atlas) return;
    const nice = (s) => s.replace(/^t slim/, "t:slim").replace(/(\d{3}) (\d{3})/, "$1/$2")
      .replace(/^not recorded$/, "(not recorded)");
    const levels = () => {
      const k = kindSel.value, sp = splitSel.value;
      const list = R.atlas.filter((x) => x.kind === k && x.split === sp);
      levelSel.innerHTML = list.map((x) => `<option value="${x.file}">${nice(x.level)}</option>`).join("");
      levelSel.disabled = list.length <= 1 && k === "multiday";
      show();
    };
    const show = () => {
      const f = levelSel.value;
      if (!f) return;
      const src = "static/atlas/" + f;
      if (frame.dataset.loaded === "1" || frame.dataset.lazy !== "1") frame.src = src;
      frame.dataset.src = src;
      open.href = src;
    };
    kindSel.addEventListener("change", levels);
    splitSel.addEventListener("change", levels);
    levelSel.addEventListener("change", show);
    levels();
    // load the iframe only when it scrolls into view (4 MB of plotly)
    const load = () => { frame.dataset.loaded = "1"; frame.src = frame.dataset.src; };
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((e) => {
        if (e[0].isIntersecting) { load(); io.disconnect(); }
      }, { rootMargin: "300px" });
      io.observe(frame);
    } else load();
  }

  /* -------------------------------------------------------- lightbox ----- */
  function initLightbox() {
    const lb = $("#lightbox"), img = $("#lightbox img");
    if (!lb) return;
    $$(".figure img").forEach((el) => el.addEventListener("click", () => {
      img.src = el.src; img.alt = el.alt; lb.classList.add("open");
    }));
    lb.addEventListener("click", () => lb.classList.remove("open"));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") lb.classList.remove("open"); });
  }

  function initCopy() {
    const btn = $("#copyBib");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const txt = $("#bibtex").textContent;
      try { await navigator.clipboard.writeText(txt); btn.textContent = "Copied"; }
      catch (e) {
        const r = document.createRange(); r.selectNodeContents($("#bibtex"));
        const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = "Selected";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    applyConfig();
    initTheme();
    initNav();
    initCounters();
    initHeroBars();
    initLive();
    initArena();
    initLoso();
    initAtlas();
    initLightbox();
    initCopy();
    const y = $("#year");
    if (y) y.textContent = new Date().getFullYear();
  });
})();
