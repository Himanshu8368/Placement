/* Site behaviour for the DSA book. Inlined into every page by scripts/gen_book.py (site_after.html),
   after window.DSA_PAGES = { "dp/229_dp_intro.html": {num, volume, volumeTitle, part, partTitle}, ... }.
   1. Real chapter numbers   2. Animation player   3. 3D tilt   4. Practice tracker
   5. Prerequisites / next   6. Quiz mode   7. Problem browser   8. Progress export / import
   9. Accessibility fixes    10. Run Python in the browser (Pyodide, loaded on first click)
   11. Continue reading on the home page                                                       */
(function () {
  "use strict";
  const PAGES = window.DSA_PAGES || {};
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function pageKey(href) {
    try {
      const u = new URL(href, location.href);
      const parts = u.pathname.split("/").filter(Boolean);
      return parts.slice(-2).join("/");          // "<folder>/<file>.html"
    } catch (e) { return ""; }
  }

  /* ---------- 1. Chapter numbers: Quarto numbers chapters in the order they exist; the book uses
     the global numbers from outline.yml. Build quartoNumber -> bookNumber from the sidebar. */
  function renumber() {
    const map = {};
    document.querySelectorAll("#quarto-sidebar a.sidebar-link").forEach(function (a) {
      const info = PAGES[pageKey(a.getAttribute("href"))];
      const num = a.querySelector(".chapter-number");
      if (info && num) {
        map[num.textContent.trim()] = String(info.num);
        num.textContent = info.num;
      }
    });
    const swap = function (s) {
      return s.replace(/^(\d+)(?=\.|$)/, function (m) { return map[m] || m; });
    };
    document.querySelectorAll(".chapter-number, .header-section-number, .toc-section-number").forEach(function (el) {
      if (el.closest("#quarto-sidebar")) return;
      el.textContent = swap(el.textContent.trim());
    });
    // "Figure 5.3", "Section 5.2.1", "Table 5.1" in captions and cross-references
    const rx = /\b(Figure|Section|Table|Chapter|Equation)( |\s)(\d+)(?=[. \s]|$)/g;
    document.querySelectorAll("figcaption, a.quarto-xref, .quarto-float-caption, .figure-caption").forEach(function (el) {
      el.childNodes.forEach(function walk(node) {
        if (node.nodeType === 3) {
          node.textContent = node.textContent.replace(rx, function (all, kind, sp, n) { return kind + sp + (map[n] || n); });
        } else if (node.childNodes) node.childNodes.forEach(walk);
      });
    });
    const info = PAGES[pageKey(location.href)];
    const header = document.querySelector("#title-block-header .quarto-title");
    if (info && header && !document.querySelector(".chapter-badge")) {
      const b = document.createElement("div");
      b.className = "chapter-badge";
      b.innerHTML = "<span>Chapter " + info.num + "</span><span>Volume " + info.volume + " · " +
        info.volumeTitle + "</span><span>Part " + info.part + " · " + info.partTitle + "</span>" +
        (info.minutes ? "<span>" + info.minutes + " min read</span>" : "") +
        (info.status ? '<span class="status st-' + info.status + '">' + info.status + "</span>" : "");
      header.prepend(b);
    }
    // Browser tab and search titles: "12 Title" -> "229 Title"
    document.title = document.title.replace(/^(\d+)(?=\s)/, function (m) { return map[m] || m; });
  }

  /* ---------- 2. Animation player for V.animate output. */
  function setupAnimations() {
    document.querySelectorAll(".dsa-anim").forEach(function (box) {
      const themes = box.querySelectorAll(".anim-theme");
      const caps = box.querySelectorAll(".anim-caption");
      const n = themes.length ? themes[0].querySelectorAll("img").length : 0;
      if (!n) return;
      const interval = parseInt(box.dataset.interval || "900", 10);
      let i = 0, timer = null;
      const bar = document.createElement("div");
      bar.className = "anim-controls";
      bar.innerHTML = '<button class="anim-btn" data-a="prev" aria-label="Previous frame">&#9664;</button>' +
        '<button class="anim-btn anim-play" data-a="play" aria-label="Play">&#9654;</button>' +
        '<button class="anim-btn" data-a="next" aria-label="Next frame">&#9654;&#9654;</button>' +
        '<div class="anim-dots"></div><span class="anim-count"></span>';
      box.appendChild(bar);
      const dots = bar.querySelector(".anim-dots");
      for (let k = 0; k < n; k++) {
        const d = document.createElement("button");
        d.className = "anim-dot"; d.setAttribute("aria-label", "Frame " + (k + 1));
        d.addEventListener("click", function () { stop(); show(k); });
        dots.appendChild(d);
      }
      function show(k) {
        i = (k + n) % n;
        themes.forEach(function (t) { t.querySelectorAll("img").forEach(function (im, j) { im.classList.toggle("on", j === i); }); });
        caps.forEach(function (c, j) { c.classList.toggle("on", j === i); });
        dots.querySelectorAll(".anim-dot").forEach(function (d, j) { d.classList.toggle("on", j === i); });
        bar.querySelector(".anim-count").textContent = (i + 1) + " / " + n;
      }
      function play() {
        if (timer) return;
        bar.querySelector(".anim-play").innerHTML = "&#10074;&#10074;";
        timer = setInterval(function () {
          if (i === n - 1) { stop(); setTimeout(function () { show(0); play(); }, 1600); return; }
          show(i + 1);
        }, interval);
      }
      function stop() {
        clearInterval(timer); timer = null;
        bar.querySelector(".anim-play").innerHTML = "&#9654;";
      }
      bar.addEventListener("click", function (e) {
        const a = e.target.closest("[data-a]");
        if (!a) return;
        if (a.dataset.a === "play") { timer ? stop() : play(); }
        else { stop(); show(i + (a.dataset.a === "next" ? 1 : -1)); }
      });
      show(0);
      if (!reduceMotion && "IntersectionObserver" in window) {
        new IntersectionObserver(function (es) {
          es.forEach(function (en) { if (en.isIntersecting && en.intersectionRatio > 0.6) play(); else stop(); });
        }, { threshold: [0, 0.6] }).observe(box);
      }
    });
  }

  /* ---------- 3. Subtle 3D tilt on cards. */
  function setupTilt() {
    if (reduceMotion || !window.matchMedia("(hover: hover)").matches) return;
    document.querySelectorAll(".tilt, .vol-card, .problem-card").forEach(function (el) {
      el.addEventListener("mousemove", function (e) {
        const r = el.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform = "perspective(900px) rotateX(" + (-y * 7).toFixed(2) + "deg) rotateY(" +
          (x * 9).toFixed(2) + "deg) translateY(-4px)";
        el.style.setProperty("--glare-x", (x + 0.5) * 100 + "%");
        el.style.setProperty("--glare-y", (y + 0.5) * 100 + "%");
      });
      el.addEventListener("mouseleave", function () { el.style.transform = ""; });
    });
  }

  /* ---------- 4. Practice tracker: a checkbox per practice link, kept in localStorage. */
  function setupTracker() {
    const KEY = "dsa-practice-done";
    let done = {};
    try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { done = {}; }
    const save = function () { try { localStorage.setItem(KEY, JSON.stringify(done)); } catch (e) { } };
    const onIndex = /problem_index\.html$/.test(location.pathname);
    const lists = document.querySelectorAll(onIndex ? "main table tbody" : ".practice ul");
    lists.forEach(function (list) {
      const items = list.querySelectorAll(":scope > li, :scope > tr");
      let counter = null;
      const container = list.closest(".practice");
      if (container) {
        counter = document.createElement("span");
        counter.className = "practice-count";
        container.appendChild(counter);
      }
      const refresh = function () {
        if (!counter) return;
        let total = 0, ok = 0;
        items.forEach(function (it) { const c = it.querySelector("input.practice-check"); if (c) { total++; if (c.checked) ok++; } });
        counter.textContent = ok + " / " + total + " solved";
        counter.classList.toggle("all", total > 0 && ok === total);
      };
      items.forEach(function (it) {
        const a = it.querySelector("a[href^='http']");
        if (!a) return;
        const id = a.href.replace(/\/$/, "");
        const box = document.createElement("input");
        box.type = "checkbox"; box.className = "practice-check"; box.title = "Mark as solved (saved in this browser)";
        box.checked = !!done[id];
        box.addEventListener("change", function () {
          if (box.checked) done[id] = Date.now(); else delete done[id];
          save(); refresh();
          document.querySelectorAll("input.practice-check").forEach(function (other) {
            if (other !== box && other.dataset.id === id) other.checked = box.checked;
          });
        });
        box.dataset.id = id;
        const cell = it.tagName === "TR" ? it.querySelector("td") : it;
        cell.prepend(box);
      });
      refresh();
    });
  }

  /* ---------- 5. Prerequisites at the top of a chapter, the next chapter at the bottom. */
  function setupFlow() {
    const info = PAGES[pageKey(location.href)];
    if (!info) return;
    const rel = function (href) { return "../" + href; };     // pages live one folder deep
    const header = document.querySelector("#title-block-header");
    if (header && info.needs && info.needs.length && !document.querySelector(".chapter-needs")) {
      const d = document.createElement("div");
      d.className = "chapter-needs";
      d.innerHTML = "<b>Read first:</b> " + info.needs.map(function (n) {
        return n.href ? '<a href="' + rel(n.href) + '">' + n.num + " · " + n.title + "</a>" : n.num + " · " + n.title;
      }).join(", ");
      header.appendChild(d);
    }
    const main = document.querySelector("main#quarto-document-content");
    if (main && info.next && info.next.href && !document.querySelector(".chapter-next")) {
      const a = document.createElement("a");
      a.className = "chapter-next tilt";
      a.href = rel(info.next.href);
      a.innerHTML = "<span>Next chapter</span><b>" + info.next.num + " · " + info.next.title + "</b>";
      main.insertBefore(a, main.querySelector("#quarto-appendix"));
    }
  }

  /* ---------- 6. Quiz mode on the question bank: one random question at a time. */
  function setupQuiz() {
    const bar = document.querySelector(".quiz-bar");
    if (!bar) return;
    const items = Array.prototype.slice.call(document.querySelectorAll(".qb-item"));
    const heads = document.querySelectorAll(".qb-chapter, .cs-link");
    const start = bar.querySelector(".quiz-start"), got = bar.querySelector(".quiz-got"),
      again = bar.querySelector(".quiz-again"), stop = bar.querySelector(".quiz-stop"),
      status = bar.querySelector(".quiz-status"), scope = bar.querySelector(".quiz-scope");
    // Light spaced repetition: each question sits in a box 0..4. "Again" sends it to box 0,
    // "Got it" moves it up one. A quiz asks the lowest boxes first, shuffled within a box.
    const KEY = "dsa-quiz-boxes";
    let boxes = {};
    try { boxes = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { boxes = {}; }
    const save = function () { try { localStorage.setItem(KEY, JSON.stringify(boxes)); } catch (e) { } };
    const idOf = function (it) {
      const q = it.querySelector("strong");
      return it.dataset.chapter + ":" + (q ? q.textContent.trim().slice(0, 80) : "");
    };
    const chapters = [];
    items.forEach(function (it) { if (chapters.indexOf(it.dataset.chapter) < 0) chapters.push(it.dataset.chapter); });
    chapters.forEach(function (c) {
      const o = document.createElement("option");
      o.value = c; o.textContent = "Chapter " + c;
      scope.appendChild(o);
    });
    let order = [], k = 0;
    const show = function () {
      items.forEach(function (it) { it.hidden = true; });
      const it = order[k];
      it.hidden = false;
      it.querySelectorAll(".callout-collapse.show, .collapse.show").forEach(function (c) { c.classList.remove("show"); });
      const learned = order.filter(function (x) { return (boxes[idOf(x)] || 0) >= 3; }).length;
      status.textContent = "Question " + (k + 1) + " of " + order.length + " · chapter " + it.dataset.chapter +
        " · box " + (boxes[idOf(it)] || 0) + " · " + learned + " learned";
      it.scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
    };
    start.addEventListener("click", function () {
      order = items.filter(function (it) { return !scope.value || it.dataset.chapter === scope.value; });
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t;
      }
      order.sort(function (a, b) { return (boxes[idOf(a)] || 0) - (boxes[idOf(b)] || 0); });   // stable: keeps the shuffle
      if (!order.length) return;
      k = 0;
      heads.forEach(function (h) { h.hidden = true; });
      document.body.classList.add("quiz-on");
      start.hidden = true; got.hidden = false; again.hidden = false; stop.hidden = false; scope.disabled = true;
      show();
    });
    const answer = function (ok) {
      const id = idOf(order[k]);
      boxes[id] = ok ? Math.min(4, (boxes[id] || 0) + 1) : 0;
      save();
      k = (k + 1) % order.length;
      show();
    };
    got.addEventListener("click", function () { answer(true); });
    again.addEventListener("click", function () { answer(false); });
    stop.addEventListener("click", function () {
      items.forEach(function (it) { it.hidden = false; });
      heads.forEach(function (h) { h.hidden = false; });
      document.body.classList.remove("quiz-on");
      start.hidden = false; got.hidden = true; again.hidden = true; stop.hidden = true;
      scope.disabled = false; status.textContent = "";
    });
  }

  /* ---------- 11. Continue reading: remember the last chapter opened; offer it on the home page. */
  function setupContinue() {
    const KEY = "dsa-last-chapter";
    const here = pageKey(location.href);
    if (PAGES[here]) {
      try { localStorage.setItem(KEY, here); } catch (e) { }
      return;
    }
    const lead = document.querySelector(".home-lead");
    let last = null;
    try { last = localStorage.getItem(KEY); } catch (e) { }
    const info = last && PAGES[last];
    if (!lead || !info) return;
    const a = document.createElement("a");
    a.className = "chapter-next continue-reading tilt";
    a.href = last;                                   // the home page sits at the site root
    a.innerHTML = "<span>Continue reading</span><b>" + info.num + " · " + info.title + "</b>";
    lead.after(a);
  }

  /* ---------- 7. Problem browser: filter the problem index tables. */
  function setupBrowser() {
    const bar = document.querySelector(".pb-bar");
    if (!bar) return;
    const rows = document.querySelectorAll("main table tbody tr");
    const q = bar.querySelector(".pb-q"), role = bar.querySelector(".pb-role"), mine = bar.querySelector(".pb-mine"),
      lo = bar.querySelector(".pb-min"), hi = bar.querySelector(".pb-max"), count = bar.querySelector(".pb-count");
    const apply = function () {
      const text = q.value.trim().toLowerCase();
      const min = parseInt(lo.value, 10), max = parseInt(hi.value, 10);
      let shown = 0;
      rows.forEach(function (tr) {
        const c = tr.cells;
        const level = parseInt(c[1] ? c[1].textContent : "", 10);
        const box = tr.querySelector("input.practice-check");
        let ok = !text || tr.textContent.toLowerCase().indexOf(text) >= 0;
        if (ok && role.value) ok = !!c[2] && c[2].textContent.trim() === role.value;
        if (ok && mine.value) ok = !!box && (box.checked === (mine.value === "done"));
        if (ok && !isNaN(min)) ok = !isNaN(level) && level >= min;
        if (ok && !isNaN(max)) ok = !isNaN(level) && level <= max;
        tr.hidden = !ok;
        if (ok) shown++;
      });
      count.textContent = shown + " of " + rows.length + " problems";
    };
    [q, role, mine, lo, hi].forEach(function (el) { el.addEventListener("input", apply); });
    document.addEventListener("change", function (e) { if (e.target.classList.contains("practice-check")) apply(); });
    apply();
  }

  /* ---------- 8. Progress export / import (the practice tracker lives in localStorage). */
  function setupProgress() {
    const bar = document.querySelector(".pb-bar");
    if (!bar) return;
    const KEY = "dsa-practice-done";
    const exp = document.createElement("button"), imp = document.createElement("button");
    exp.type = "button"; imp.type = "button";
    exp.textContent = "Export progress"; imp.textContent = "Import progress";
    const file = document.createElement("input");
    file.type = "file"; file.accept = "application/json"; file.hidden = true;
    exp.addEventListener("click", function () {
      let data = "{}";
      try { data = localStorage.getItem(KEY) || "{}"; } catch (e) { }
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([data], { type: "application/json" }));
      a.download = "dsa-progress.json";
      a.click();
    });
    imp.addEventListener("click", function () { file.click(); });
    file.addEventListener("change", function () {
      const f = file.files[0];
      if (!f) return;
      f.text().then(function (t) {
        let incoming = {}, cur = {};
        try { incoming = JSON.parse(t); cur = JSON.parse(localStorage.getItem(KEY) || "{}"); }
        catch (e) { alert("Not a progress file."); return; }
        Object.keys(incoming).forEach(function (k) { if (!cur[k]) cur[k] = incoming[k]; });   // merge, never drop
        try { localStorage.setItem(KEY, JSON.stringify(cur)); } catch (e) { }
        location.reload();
      });
    });
    bar.append(exp, imp, file);
  }

  /* ---------- 9. Accessibility: alt text from the caption; keyboard control of animations. */
  function setupA11y() {
    document.querySelectorAll("figure img:not([alt]), figure img[alt='']").forEach(function (img) {
      const cap = img.closest("figure").querySelector("figcaption");
      if (cap) img.alt = cap.textContent.trim();
    });
    document.querySelectorAll(".dsa-anim").forEach(function (box) {
      box.addEventListener("keydown", function (e) {
        const a = { ArrowLeft: "prev", ArrowRight: "next", " ": "play" }[e.key];
        if (!a) return;
        e.preventDefault();
        const btn = box.querySelector('[data-a="' + a + '"]');
        if (btn) btn.click();
      });
    });
  }

  /* ---------- 10. "Run" on Python listings: Pyodide is fetched only on the first click. */
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  let pyReady = null;
  function loadPy() {
    if (pyReady) return pyReady;
    pyReady = new Promise(function (ok, fail) {
      const s = document.createElement("script");
      s.src = PYODIDE + "pyodide.js";
      s.onload = function () { window.loadPyodide({ indexURL: PYODIDE }).then(ok, fail); };
      s.onerror = fail;
      document.head.appendChild(s);
    });
    return pyReady;
  }
  function setupRun() {
    document.querySelectorAll("div.sourceCode pre.sourceCode.python").forEach(function (pre) {
      const code = pre.querySelector("code");
      // Drawings and trace boards need the build's modules; leave those listings alone.
      if (!code || /dsa_viz|matplotlib|trace|import V\b|sys\.path/.test(code.textContent)) return;
      const box = pre.closest("div.sourceCode");
      const btn = document.createElement("button");
      btn.type = "button"; btn.className = "run-btn"; btn.textContent = "Run";
      btn.setAttribute("aria-label", "Run this Python code in the browser");
      const out = document.createElement("pre");
      out.className = "run-out"; out.hidden = true; out.setAttribute("aria-live", "polite");
      box.after(out);
      box.appendChild(btn);
      btn.addEventListener("click", function () {
        btn.disabled = true; out.hidden = false; out.textContent = "Loading Python (first run only)…";
        loadPy().then(function (py) {
          const buf = [];
          py.setStdout({ batched: function (t) { buf.push(t); } });
          py.setStderr({ batched: function (t) { buf.push(t); } });
          const src = "import random, math, sys, itertools, functools, collections, heapq, bisect\n" +
            "from collections import *\nfrom functools import *\nfrom itertools import *\n" + code.textContent;
          return py.runPythonAsync(src).then(function () {
            out.textContent = buf.join("\n") || "(no output)";
          }, function (err) {
            out.textContent = buf.join("\n") + "\n" + String(err).split("\n").slice(-3).join("\n");
          });
        }, function () { out.textContent = "Could not load Pyodide (offline?)."; })
          .then(function () { btn.disabled = false; });
      });
    });
  }

  function init() {
    renumber(); setupAnimations(); setupTilt(); setupTracker(); setupFlow(); setupQuiz();
    setupBrowser(); setupProgress(); setupA11y(); setupRun(); setupContinue();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
