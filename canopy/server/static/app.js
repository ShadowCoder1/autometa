/* Canopy — the review UI.
 *
 * Vanilla JS, no build step, no network except this server. One rule governs the whole file:
 * every string that came out of a PDF or a model is written with `textContent`, never as markup.
 * The DOM is built with the `h()` helper below for the same reason — there is no path in this
 * file by which a quote, a citation, a model's reason or a file name could become HTML.
 *
 * The forest plot is the one exception that needs care: it is an SVG the server generated, and it
 * is parsed with DOMParser, stripped of anything executable, and only then adopted into the page.
 */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  /* ───────────────────────────────────────────────────────── DOM helpers */
  function h(tag, opts, kids) {
    var node = document.createElement(tag);
    opts = opts || {};
    if (opts.cls) { node.className = opts.cls; }
    if (opts.text !== undefined && opts.text !== null) { node.textContent = String(opts.text); }
    if (opts.attrs) {
      Object.keys(opts.attrs).forEach(function (key) {
        var value = opts.attrs[key];
        if (value !== null && value !== undefined && value !== false) {
          node.setAttribute(key, String(value));
        }
      });
    }
    if (opts.on) {
      Object.keys(opts.on).forEach(function (name) { node.addEventListener(name, opts.on[name]); });
    }
    (kids || []).forEach(function (kid) { if (kid) { node.appendChild(kid); } });
    return node;
  }
  function $(id) { return document.getElementById(id); }
  function clear(node) { while (node && node.firstChild) { node.removeChild(node.firstChild); } }
  function show(node, on) { if (node) { node.hidden = !on; } }

  function num(value, digits) {
    if (value === null || value === undefined || value === "" || isNaN(Number(value))) { return "—"; }
    return Number(value).toFixed(digits === undefined ? 3 : digits);
  }
  function money(value) { return "$" + (Number(value) || 0).toFixed(2); }
  function lines(text) {
    return String(text || "").split("\n").map(function (s) { return s.trim(); })
      .filter(function (s) { return s.length > 0; });
  }
  function commas(text) {
    return String(text || "").split(",").map(function (s) { return s.trim(); })
      .filter(function (s) { return s.length > 0; });
  }

  /* ───────────────────────────────────────────────────────── state + API */
  var state = {
    settings: null, examples: [], runId: "", token: "", run: null, results: null,
    outcome: "", files: [], events: [], papers: {}, source: null, mode: "guided",
    started: false, returnFocus: null, live: {}, poll: null, names: {}
  };

  /* The run a reload must not lose. Tokens are kept per run id (so any run this browser
     started can be re-opened); `current` is the one the page re-attaches to. */
  var STORE_TOKENS = "canopy.tokens";
  var STORE_CURRENT = "canopy.current";

  function store(key, value) {
    try {
      if (value === null) { window.localStorage.removeItem(key); }
      else { window.localStorage.setItem(key, JSON.stringify(value)); }
    } catch (err) { /* private mode, or a full disk: the UI still works, it just forgets */ }
  }

  function stored(key, fallback) {
    try {
      var raw = window.localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (err) { return fallback; }
  }

  function rememberRun(runId, token) {
    var tokens = stored(STORE_TOKENS, {}) || {};
    tokens[runId] = token;
    store(STORE_TOKENS, tokens);
    store(STORE_CURRENT, runId);
  }

  function forgetCurrentRun() { store(STORE_CURRENT, null); }

  function tokenFor(runId) { return (stored(STORE_TOKENS, {}) || {})[runId] || ""; }

  function toast(message) {
    var box = $("toast");
    box.textContent = String(message);
    show(box, true);
    window.clearTimeout(toast._t);
    toast._t = window.setTimeout(function () { show(box, false); }, 4200);
  }

  function api(path, opts) {
    opts = opts || {};
    var headers = {};
    Object.keys(opts.headers || {}).forEach(function (k) { headers[k] = opts.headers[k]; });
    if (state.token && path.indexOf("/api/runs") === 0) {
      headers.Authorization = "Bearer " + state.token;
    }
    var init = { method: opts.method || "GET", headers: headers };
    if (opts.json !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.json);
    } else if (opts.body !== undefined) {
      init.body = opts.body;
    }
    return fetch(path, init).then(function (response) {
      return response.text().then(function (text) {
        var data = null;
        if (text) { try { data = JSON.parse(text); } catch (err) { data = { detail: text }; } }
        if (!response.ok) {
          var detail = (data && data.detail) ? data.detail : (response.status + " " + response.statusText);
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }
        return data;
      });
    });
  }

  function withToken(url) {
    if (!url) { return ""; }
    return url + (url.indexOf("?") >= 0 ? "&" : "?") + "token=" + encodeURIComponent(state.token);
  }

  /* ───────────────────────────────────────────────────────── screens */
  var SCREENS = ["new", "monitor", "results", "runs"];

  function goto(name) {
    SCREENS.forEach(function (key) { show($("screen-" + key), key === name); });
    Array.prototype.forEach.call(document.querySelectorAll(".step"), function (button) {
      var on = button.getAttribute("data-screen") === name;
      button.classList.toggle("is-current", on);
      if (on) { button.setAttribute("aria-current", "step"); } else { button.removeAttribute("aria-current"); }
    });
    window.scrollTo(0, 0);
    if (name === "runs") { loadRuns(); }
  }

  Array.prototype.forEach.call(document.querySelectorAll(".step"), function (button) {
    button.addEventListener("click", function () {
      var screen = button.getAttribute("data-screen");
      if (screen === "new") { forgetCurrentRun(); }   // the next reload starts on a clean form
      goto(screen);
    });
  });

  /* ───────────────────────────────────────────────────────── settings */
  function loadSettings() {
    return api("/api/settings").then(function (settings) {
      state.settings = settings;
      var pill = $("key-pill");
      pill.textContent = settings.api_key_configured ? "api key configured" : "no api key";
      pill.className = "pill " + (settings.api_key_configured ? "on" : "off");
      var profiles = $("p-profile");
      clear(profiles);
      (settings.profiles || []).forEach(function (name) {
        profiles.appendChild(h("option", { text: name, attrs: { value: name } }));
      });
      $("o-model-primary").value = (settings.models || {}).primary || "";
      $("o-model-secondary").value = (settings.models || {}).secondary || "";
      $("files-note").textContent = "Nothing is uploaded until you press Run. PDFs only, up to "
        + settings.max_upload_mb + " MB each.";
    });
  }

  function loadExamples() {
    return api("/api/protocols/examples").then(function (body) {
      state.examples = body.examples || [];
      var picker = $("example-picker");
      clear(picker);
      picker.appendChild(h("option", { text: "— a blank skeleton —", attrs: { value: "" } }));
      state.examples.forEach(function (example) {
        picker.appendChild(h("option", { text: example.title, attrs: { value: example.name } }));
      });
      picker.addEventListener("change", function () {
        var chosen = state.examples.filter(function (e) { return e.name === picker.value; })[0];
        $("p-yaml").value = chosen ? chosen.yaml : (body.skeleton || "");
      });
      $("p-yaml").value = body.skeleton || "";
    });
  }

  /* ───────────────────────────────────────────────────────── protocol form */
  function outcomeCard(values, index) {
    values = values || {};
    var fields = [
      ["label", "Name", "text"],
      ["key", "Key (lower_snake_case)", "text"],
      ["definition", "What counts as this outcome", "area"],
      ["measurement_window", "Which timepoint or block to read", "area"],
      ["higher_is_better_hint", "How to tell whether a larger number is more or less", "area"],
      ["positive_direction_label", "Forest axis label, positive side", "text"],
      ["negative_direction_label", "Forest axis label, negative side", "text"],
      ["units_hint", "Units you expect", "text"]
    ];
    var card = h("div", { cls: "outcome", attrs: { "data-outcome": String(index) } });
    var head = h("div", { cls: "outcome-head" }, [
      h("strong", { text: "outcome " + (index + 1) }),
      h("button", {
        cls: "btn ghost small", text: "Remove", attrs: { type: "button" },
        on: { click: function () { card.parentNode.removeChild(card); renumberOutcomes(); } }
      })
    ]);
    card.appendChild(head);
    fields.forEach(function (field) {
      var input = field[2] === "area"
        ? h("textarea", { attrs: { rows: 2, "data-field": field[0] } })
        : h("input", { attrs: { type: "text", "data-field": field[0] } });
      input.value = values[field[0]] || "";
      card.appendChild(h("label", { cls: "field" },
        [h("span", { cls: "label", text: field[1] }), input]));
    });
    return card;
  }

  function renumberOutcomes() {
    Array.prototype.forEach.call($("outcomes").children, function (card, index) {
      card.querySelector("strong").textContent = "outcome " + (index + 1);
    });
  }

  function addOutcome(values) {
    var holder = $("outcomes");
    holder.appendChild(outcomeCard(values, holder.children.length));
  }

  function collectOutcomes() {
    return Array.prototype.map.call($("outcomes").children, function (card) {
      var out = {};
      Array.prototype.forEach.call(card.querySelectorAll("[data-field]"), function (input) {
        out[input.getAttribute("data-field")] = input.value.trim();
      });
      if (!out.key) {
        out.key = (out.label || "outcome").toLowerCase().replace(/[^a-z0-9]+/g, "_")
          .replace(/^_+|_+$/g, "") || "outcome";
      }
      return out;
    });
  }

  function guidedProtocol() {
    return {
      title: $("p-title").value.trim(),
      research_question: $("p-question").value.trim(),
      group_a: {
        key: "A", label: $("a-label").value.trim(), definition: $("a-definition").value.trim(),
        synonyms: commas($("a-synonyms").value)
      },
      group_b: {
        key: "B", label: $("b-label").value.trim(), definition: $("b-definition").value.trim(),
        synonyms: commas($("b-synonyms").value)
      },
      outcomes: collectOutcomes(),
      eligibility: lines($("p-eligibility").value),
      dataset_rules: lines($("p-dataset-rules").value),
      moderators: lines($("p-moderators").value),
      stats: { profile: $("p-profile").value },
      notes: $("p-notes").value.trim()
    };
  }

  function fillGuided(protocol) {
    $("p-title").value = protocol.title || "";
    $("p-question").value = protocol.research_question || "";
    var a = protocol.group_a || {}, b = protocol.group_b || {};
    $("a-label").value = a.label || "";
    $("a-definition").value = a.definition || "";
    $("a-synonyms").value = (a.synonyms || []).join(", ");
    $("b-label").value = b.label || "";
    $("b-definition").value = b.definition || "";
    $("b-synonyms").value = (b.synonyms || []).join(", ");
    clear($("outcomes"));
    (protocol.outcomes || []).forEach(function (outcome) { addOutcome(outcome); });
    if (!(protocol.outcomes || []).length) { addOutcome(null); }
    $("p-eligibility").value = (protocol.eligibility || []).join("\n");
    $("p-dataset-rules").value = (protocol.dataset_rules || []).join("\n");
    $("p-moderators").value = (protocol.moderators || []).join("\n");
    $("p-notes").value = protocol.notes || "";
    if (protocol.stats && protocol.stats.profile) { $("p-profile").value = protocol.stats.profile; }
  }

  Array.prototype.forEach.call(document.querySelectorAll(".seg-btn"), function (button) {
    button.addEventListener("click", function () {
      state.mode = button.getAttribute("data-mode");
      Array.prototype.forEach.call(document.querySelectorAll(".seg-btn"), function (other) {
        var on = other === button;
        other.classList.toggle("is-on", on);
        other.setAttribute("aria-selected", on ? "true" : "false");
      });
      show($("guided"), state.mode === "guided");
      show($("yaml-mode"), state.mode === "yaml");
    });
  });

  $("add-outcome").addEventListener("click", function () { addOutcome(null); });

  $("draft-btn").addEventListener("click", function () {
    var sentence = $("draft-sentence").value.trim();
    if (!sentence) { toast("Say what your review is about first."); return; }
    var button = $("draft-btn");
    button.disabled = true;
    button.textContent = "Drafting…";
    api("/api/protocols/draft", { method: "POST", json: { sentence: sentence } })
      .then(function (body) {
        fillGuided(body.protocol || {});
        $("p-yaml").value = body.yaml || "";
        var note = $("draft-note");
        clear(note);
        (body.warnings || []).forEach(function (warning) {
          note.appendChild(h("span", { text: warning + " " }));
        });
        toast("Drafted — now read every line of it.");
      })
      .catch(function (error) { toast(error.message); })
      .then(function () { button.disabled = false; button.textContent = "Draft it for me"; });
  });

  /* ───────────────────────────────────────────────────────── files */
  function acceptFiles(fileList) {
    var chosen = Array.prototype.filter.call(fileList || [], function (file) {
      return /\.pdf$/i.test(file.name);
    });
    state.files = chosen;
    var list = $("file-list");
    clear(list);
    chosen.slice(0, 400).forEach(function (file) {
      list.appendChild(h("li", {}, [
        h("span", { text: file.name }),
        h("span", { text: (file.size / 1e6).toFixed(1) + " MB" })
      ]));
    });
    var skipped = (fileList ? fileList.length : 0) - chosen.length;
    $("files-note").textContent = chosen.length + " PDF" + (chosen.length === 1 ? "" : "s")
      + " ready" + (skipped > 0 ? " · " + skipped + " non-PDF file(s) ignored" : "");
  }

  $("folder-input").addEventListener("change", function (event) { acceptFiles(event.target.files); });
  $("files-input").addEventListener("change", function (event) { acceptFiles(event.target.files); });

  var drop = $("drop");
  ["dragenter", "dragover"].forEach(function (name) {
    drop.addEventListener(name, function (event) {
      event.preventDefault(); drop.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    drop.addEventListener(name, function (event) {
      event.preventDefault(); drop.classList.remove("is-over");
    });
  });
  drop.addEventListener("drop", function (event) {
    if (event.dataTransfer && event.dataTransfer.files) { acceptFiles(event.dataTransfer.files); }
  });

  /* ───────────────────────────────────────────────────────── create a run */
  function runOptions() {
    var options = { concurrency: Number($("o-concurrency").value) || 4, models: {} };
    // the picker only speaks for the guided form; a pasted protocol keeps its own stats block
    if (state.mode === "guided") { options.profile = $("p-profile").value; }
    if ($("o-budget").value) { options.budget_usd = Number($("o-budget").value); }
    if ($("o-per-paper").value) { options.max_usd_per_paper = Number($("o-per-paper").value); }
    if ($("o-max-papers").value) { options.max_papers = Number($("o-max-papers").value); }
    var primary = $("o-model-primary").value.trim(), secondary = $("o-model-secondary").value.trim();
    if (primary) { options.models.primary = primary; }
    if (secondary) { options.models.secondary = secondary; }
    return options;
  }

  function buildForm(start) {
    var form = new FormData();
    state.files.forEach(function (file) { form.append("files", file, file.name); });
    if (state.mode === "yaml") {
      form.append("protocol_text", $("p-yaml").value);
    } else {
      form.append("protocol_text", JSON.stringify(guidedProtocol()));
    }
    var options = runOptions();
    options.start = !!start;
    form.append("options", JSON.stringify(options));
    return form;
  }

  // The create request carries every PDF, so it can take minutes on a large folder. `fetch` gives
  // no upload progress, and a button that greys out for three minutes with nothing else on screen
  // reads as a hang — so this one request goes through XHR and reports what it is doing.
  function postRun(form) {
    return new Promise(function (resolve, reject) {
      var request = new XMLHttpRequest();
      request.open("POST", "/api/runs");
      request.upload.addEventListener("progress", function (event) {
        if (!event.lengthComputable) { return; }
        uploadProgress(Math.round((event.loaded / event.total) * 100));
      });
      request.upload.addEventListener("load", function () {
        uploadProgress(100, "Reading the PDFs…");   // the server is probing them now
      });
      request.addEventListener("load", function () {
        var data = null;
        try { data = JSON.parse(request.responseText || "null"); } catch (err) { data = null; }
        if (request.status >= 200 && request.status < 300) { resolve(data); return; }
        var detail = (data && data.detail) ? data.detail : request.status + " " + request.statusText;
        reject(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
      });
      request.addEventListener("error", function () { reject(new Error("the upload failed")); });
      request.addEventListener("abort", function () { reject(new Error("the upload was cancelled")); });
      request.send(form);
    });
  }

  function uploadProgress(percent, message) {
    var note = $("upload-progress");
    if (!note) { return; }
    show(note, percent !== null);
    if (percent === null) { return; }
    note.textContent = message || ("Uploading " + state.files.length + " files… " + percent + "%");
  }

  function createRun(start) {
    var problem = $("form-error");
    show(problem, false);
    if (!state.files.length) { problem.textContent = "Choose a folder of PDFs first."; show(problem, true); return Promise.reject(new Error("no files")); }
    uploadProgress(0);
    return postRun(buildForm(start)).then(function (body) {
      uploadProgress(null);
      state.runId = body.run_id;
      state.token = body.token;
      state.started = !!start;
      state.papers = {};
      state.live = {};
      state.names = {};
      rememberRun(body.run_id, body.token);
      $("run-title").textContent = body.title + " · " + body.run_id;
      show($("cost-meter"), true);
      return body;
    }).catch(function (error) {
      uploadProgress(null);
      problem.textContent = error.message;
      show(problem, true);
      throw error;
    });
  }

  $("run-form").addEventListener("submit", function (event) {
    event.preventDefault();
    var button = $("run-btn");
    button.disabled = true;
    // a run that was created for a dry run is started, not created again: the papers are already
    // uploaded and its stage files are the ones `--resume` will reuse
    var promise = (state.runId && !state.started)
      ? api("/api/runs/" + state.runId + "/start", { method: "POST" }).then(function (body) {
        state.started = true;
        return body;
      })
      : createRun(true);
    promise.then(function () {
      state.papers = {};
      state.events = [];
      clear($("log"));
      clear($("stage-grid").tBodies[0]);
      goto("monitor");
      listen();
    }).catch(function () { /* the message is already on screen */ })
      .then(function () { button.disabled = false; });
  });

  $("dry-run-btn").addEventListener("click", function () {
    var button = $("dry-run-btn");
    button.disabled = true;
    button.textContent = "Mapping…";
    var ready = (state.runId && !state.started) ? Promise.resolve(null) : createRun(false);
    ready.then(function () {
      return api("/api/runs/" + state.runId + "/dry-run",
        { method: "POST", json: { max_papers: 3, wait_seconds: 0 } });
    }).then(function (body) {
      return body.status === "running" ? pollDryRun() : body;
    }).then(function (body) {
      renderDryRun(body);
    }).catch(function (error) { toast(error.message); })
      .then(function () {
        button.disabled = false;
        button.textContent = "Dry run the mapper on 2–3 papers";
      });
  });

  function pollDryRun() {
    return new Promise(function (resolve, reject) {
      var tries = 0;
      var timer = window.setInterval(function () {
        tries += 1;
        api("/api/runs/" + state.runId + "/dry-run").then(function (body) {
          if (body.status !== "running") {
            window.clearInterval(timer);
            resolve(body);
          } else if (tries > 800) {
            window.clearInterval(timer);
            reject(new Error("the dry run is taking too long — watch the log instead"));
          }
        }).catch(function (error) { window.clearInterval(timer); reject(error); });
      }, 1500);
    });
  }

  function renderDryRun(body) {
    var holder = $("dry-run-out");
    clear(holder);
    if (body.error) { holder.appendChild(h("p", { cls: "error", text: body.error })); return; }
    var card = h("section", { cls: "card" }, [
      h("div", { cls: "card-head" }, [
        h("h2", { text: "Dry run — what the mapper found" }),
        h("span", { cls: "hint", text: money(body.cost_usd) + " spent" })
      ])
    ]);
    (body.maps || []).forEach(function (study) {
      var citation = study.citation || {};
      var head = h("div", { cls: "flag-head" }, [
        h("strong", { text: (citation.first_author || citation.authors || study.filename || "paper")
          + (citation.year ? " " + citation.year : "") }),
        h("span", { cls: "badge " + (study.eligible ? "ok" : "stop"),
          text: study.eligible ? "eligible" : "not eligible" })
      ]);
      var body_ = h("div", {}, [
        h("p", { cls: "hint", text: study.eligibility_rationale || study.exclusion_reason || "" })
      ]);
      (study.datasets || []).forEach(function (dataset) {
        var groups = (dataset.group_a || {}).label + " (n=" + ((dataset.group_a || {}).n || "?") + ")"
          + " vs " + (dataset.group_b || {}).label + " (n=" + ((dataset.group_b || {}).n || "?") + ")";
        body_.appendChild(h("p", { cls: "mono", text: dataset.dataset_id + " · " + groups }));
        (dataset.outcomes || []).forEach(function (outcome) {
          var where = (outcome.sources || []).map(function (source) {
            return source.kind + " p" + source.page + (source.locator ? " " + source.locator : "");
          }).join("; ");
          body_.appendChild(h("p", { cls: "hint", text: "  " + outcome.outcome_key + " ← "
            + (where || "no source located") }));
        });
      });
      (study.needs_human || []).forEach(function (note) {
        body_.appendChild(h("p", { cls: "hint", text: "needs a human: " + note }));
      });
      card.appendChild(h("div", { cls: "flag" }, [head, body_]));
    });
    holder.appendChild(card);
  }

  /* ───────────────────────────────────────────────────────── monitor */
  var STAGES = ["ingest", "map", "extract", "verify", "resolve"];

  function stageClass(status) {
    if (status === "done") { return "dot done"; }
    if (status === "started") { return "dot run"; }
    if (status === "skipped") { return "dot skip"; }
    if (status === "excluded") { return "dot hold"; }
    if (status === "error" || status === "budget") { return "dot error"; }
    return "dot";
  }

  function paperRow(id, meta) {
    var entry = state.papers[id];
    if (!entry) {
      var cells = {};
      var row = h("tr");
      var label = h("strong", { text: id });
      var sub = h("span", { cls: "hint", text: id });
      row.appendChild(h("td", { cls: "paper" }, [label, h("br"), sub]));
      STAGES.forEach(function (stage) {
        var cell = h("td");
        var dot = h("span", { cls: "dot", text: "" });
        cell.appendChild(dot);
        cells[stage] = dot;
        row.appendChild(cell);
      });
      var cost = h("td", { cls: "num", text: "—" });
      row.appendChild(cost);
      $("stage-grid").tBodies[0].appendChild(row);
      entry = { row: row, cells: cells, cost: cost, label: label, sub: sub };
      state.papers[id] = entry;
    }
    if (meta) {
      entry.label.textContent = meta.study_label || meta.filename || id;
      entry.sub.textContent = meta.filename ? meta.filename + " · " + id : id;
      entry.row.setAttribute("title", meta.paper_id || id);
      if (meta.error) { entry.row.setAttribute("title", meta.error); }
    }
    return entry;
  }

  function applyStage(entry, stage, status, message) {
    var dot = entry.cells[stage];
    if (!dot) { return; }
    dot.className = stageClass(status);
    dot.textContent = status === "skipped" ? "cached" : "";
    if (message) { dot.setAttribute("title", message); }
  }

  /* The manifest is the truth about a finished run: a monitor opened after the fact — or after a
     reload — paints the same grid the live events would have drawn. Events are an overlay on top
     of it, which is what keeps a running paper's ◐ from being erased by a refresh. */
  function renderPapers(papers) {
    (papers || []).forEach(function (paper) {
      var id = paper.sha12 || String(paper.paper_id || "").slice(0, 12);
      if (!id) { return; }
      var entry = paperRow(id, paper);
      STAGES.forEach(function (stage) {
        var status = (paper.stages || {})[stage];
        if (status) { applyStage(entry, stage, status, ""); }
      });
      if (paper.status === "error" || paper.status === "cancelled") {
        STAGES.forEach(function (stage) {
          if (!(paper.stages || {})[stage]) {
            applyStage(entry, stage, paper.status, paper.error || paper.status);
          }
        });
      }
      entry.cost.textContent = money(paper.cost_usd);
    });
    Object.keys(state.live).forEach(function (id) {           // live events win over the manifest
      var entry = state.papers[id];
      if (!entry) { return; }
      Object.keys(state.live[id]).forEach(function (stage) {
        var seen = state.live[id][stage];
        if (seen.status === "started") { applyStage(entry, stage, "started", seen.message); }
      });
    });
    $("stage-summary").textContent = Object.keys(state.papers).length + " paper(s)";
  }

  function onEvent(event) {
    state.events.push(event);
    if (event.cost_so_far !== undefined) {
      $("cost-value").textContent = money(event.cost_so_far);
      show($("cost-meter"), true);
    }
    if (event.paper && STAGES.indexOf(event.stage) >= 0) {
      state.live[event.paper] = state.live[event.paper] || {};
      state.live[event.paper][event.stage] = { status: event.status, message: event.message };
      applyStage(paperRow(event.paper, state.names[event.paper]), event.stage, event.status,
                 event.message);
    }
    if (event.paper && event.stage === "paper") {
      var failed = paperRow(event.paper, state.names[event.paper]);
      STAGES.forEach(function (stage) {
        if (!failed.cells[stage].className.match(/done|skip/)) {
          failed.cells[stage].className = "dot error";
        }
      });
    }
    var log = $("log");
    var who = (state.names[event.paper] || {}).study_label || event.paper || "—";
    log.appendChild(h("li", {}, [
      h("b", { text: event.stage }),
      h("span", { cls: "who", text: who, attrs: { title: event.paper || "" } }),
      h("span", { cls: "msg", text: (event.status || "") + (event.message ? " · " + event.message : "") })
    ]));
    while (log.children.length > 400) { log.removeChild(log.firstChild); }
    log.scrollTop = log.scrollHeight;
    $("log-count").textContent = state.events.length + " events";
    $("stage-summary").textContent = Object.keys(state.papers).length + " paper(s)";
  }

  function listen() {
    if (state.source) { state.source.close(); }
    var source = new EventSource(withToken("/api/runs/" + state.runId + "/events"));
    state.source = source;
    source.addEventListener("progress", function (message) {
      try { onEvent(JSON.parse(message.data)); } catch (err) { /* a frame we cannot read */ }
    });
    source.addEventListener("end", function (message) {
      var event = {};
      try { event = JSON.parse(message.data); } catch (err) { event = {}; }
      onEvent(event);
      source.close();
      state.source = null;
      $("monitor-sub").textContent = (event.status === "not_started"
        ? "This run has not been started yet."
        : "Run " + (event.status || "finished") + (event.message ? " — " + event.message : ""));
      refreshRun().then(function () {
        if (event.status === "done") { loadResults(); }
      }).catch(function (error) { toast(error.message); });
    });
    source.onerror = function () { source.close(); state.source = null; };
  }

  $("cancel-btn").addEventListener("click", function () {
    if (!state.runId) {          // the create request is still uploading: there is nothing to cancel
      toast("The papers are still uploading — the run has not been created yet. "
            + "Reload the page to abandon it.");
      return;
    }
    api("/api/runs/" + state.runId + "/cancel", { method: "POST" })
      .then(function (body) { toast("Run " + body.status + " — what finished is kept."); })
      .catch(function (error) { toast(error.message); });
  });

  /* ───────────────────────────────────────────────────────── run state */
  var TERMINAL = ["done", "error", "cancelled", "interrupted"];

  function refreshRun() {
    if (!state.runId) { return Promise.resolve(null); }
    return api("/api/runs/" + state.runId).then(function (run) {
      state.run = run;
      rememberRun(state.runId, state.token);
      $("run-title").textContent = (run.title || run.run_id) + " · " + run.status;
      $("cost-value").textContent = money(run.cost_usd);
      show($("cost-meter"), true);
      var picker = $("outcome-picker");
      clear(picker);
      (run.outcomes || []).forEach(function (outcome) {
        picker.appendChild(h("option", {
          text: outcome.label + (outcome.k ? " (k=" + outcome.k + ")" : ""),
          attrs: { value: outcome.key }
        }));
      });
      if (!state.outcome && (run.outcomes || []).length) { state.outcome = run.outcomes[0].key; }
      picker.value = state.outcome;
      (run.papers || []).forEach(function (paper) {
        state.names[paper.sha12 || String(paper.paper_id || "").slice(0, 12)] = paper;
      });
      renderPapers(run.papers || []);
      showRunState(run);
      return run;
    });
  }

  function showRunState(run) {
    var finished = TERMINAL.indexOf(run.status) >= 0;
    var stop = $("cancel-btn");
    stop.disabled = finished;
    stop.textContent = finished ? "Run " + run.status : "Stop the run";
    if (finished) { stop.setAttribute("title", "this run has already finished"); }
    else { stop.removeAttribute("title"); }

    var banner = $("monitor-banner");
    clear(banner);
    show(banner, finished);
    if (finished) {
      var word = run.status === "done" ? "Run finished." : "Run " + run.status + ".";
      banner.appendChild(h("span", { text: word + " " + money(run.cost_usd) + " over "
        + (run.papers || []).length + " paper(s)." }));
      if (run.status !== "error") {
        banner.appendChild(h("button", {
          cls: "btn primary small", text: "Open Results", attrs: { type: "button" },
          on: { click: function () { loadResults(); } }
        }));
      }
    }
    pollWhileRunning(run);
  }

  /* A run outlives the page that started it: while one is going the monitor asks the server for
     the manifest every few seconds, so cost and stages are right even if the event stream dropped
     or the tab was asleep. */
  function pollWhileRunning(run) {
    var going = TERMINAL.indexOf(run.status) < 0;
    if (going && !state.poll) {
      state.poll = window.setInterval(function () {
        refreshRun().catch(function () { /* a transient failure is not worth a toast */ });
      }, 5000);
    }
    if (!going && state.poll) {
      window.clearInterval(state.poll);
      state.poll = null;
    }
  }

  function attach(runId, token) {
    state.runId = runId;
    state.token = token;
    state.outcome = "";
    state.papers = {};
    state.live = {};
    state.names = {};
    clear($("stage-grid").tBodies[0]);
    clear($("log"));
    state.events = [];
    return refreshRun().then(function (run) {
      state.started = run.status !== "created";
      rememberRun(runId, token);
      if (TERMINAL.indexOf(run.status) < 0) {
        goto("monitor");
        listen();
      } else if ((run.outcomes || []).some(function (o) { return o.has_results; })) {
        loadResults();
      } else {
        goto("monitor");
      }
      return run;
    });
  }

  $("outcome-picker").addEventListener("change", function (event) {
    state.outcome = event.target.value;
    loadResults();
  });

  /* ───────────────────────────────────────────────────────── results */
  Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (tab) {
    tab.addEventListener("click", function () {
      var name = tab.getAttribute("data-tab");
      Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (other) {
        var on = other === tab;
        other.classList.toggle("is-on", on);
        other.setAttribute("aria-selected", on ? "true" : "false");
      });
      ["forest", "table", "flags", "figures", "downloads"].forEach(function (key) {
        show($("pane-" + key), key === name);
      });
    });
  });

  $("repool-btn").addEventListener("click", function () {
    var button = $("repool-btn");
    button.disabled = true;
    api("/api/runs/" + state.runId + "/repool", { method: "POST" })
      .then(function (summary) {
        toast(summary.applied + " override(s) applied"
          + (summary.pending.length ? ", " + summary.pending.length + " need a re-run" : ""));
        highlightRepool(false);
        return refreshRun().then(loadResults);
      })
      .catch(function (error) { toast(error.message); })
      .then(function () { button.disabled = false; });
  });

  function loadResults() {
    if (!state.runId || !state.outcome) { return Promise.resolve(null); }
    return api("/api/runs/" + state.runId + "/results/" + encodeURIComponent(state.outcome))
      .then(function (results) {
        state.results = results;
        goto("results");
        renderPooled(results);
        highlightRepool(false);
        renderTable(results);
        renderFlags(results);
        renderFigures(results);
        renderDownloads(results);
        return renderForest(results);
      })
      .catch(function (error) { toast(error.message); });
  }

  function renderPooled(results) {
    var pooled = results.pooled || {};
    var outcome = results.outcome || {};
    var settings = results.settings || {};
    $("results-title").textContent = outcome.label || state.outcome;
    $("results-sub").textContent = outcome.definition || "";
    var card = $("pooled-card");
    clear(card);
    if (!pooled.k) {
      card.appendChild(h("p", { text: pooled.note || "Nothing could be pooled for this outcome." }));
      return;
    }
    function stat(key, value, small) {
      return h("div", { cls: "stat" }, [
        h("span", { cls: "k", text: key }),
        h("span", { cls: "v" }, [document.createTextNode(value),
          small ? h("small", { text: " " + small }) : null])
      ]);
    }
    card.appendChild(stat("pooled estimate", num(pooled.estimate)));
    card.appendChild(stat("95% CI", "[" + num(pooled.ci_low, 2) + ", " + num(pooled.ci_high, 2) + "]"));
    card.appendChild(stat("k", String(pooled.k), "from " + (pooled.k_papers || pooled.k) + " papers"));
    card.appendChild(stat("I²", num(pooled.I2, 1) + "%"));
    card.appendChild(stat("τ²", num(pooled.tau2, 3)));
    card.appendChild(stat("held for review", String(pooled.n_needs_human || 0)));
    card.appendChild(h("div", { cls: "stat wide" }, [
      h("span", { cls: "k", text: "conventions" }),
      h("span", { cls: "v", text: [settings.estimator, settings.variance, settings.tau2_method,
        settings.hakn ? "Hartung-Knapp" : "z intervals",
        "PI " + settings.pi_method].filter(Boolean).join(" · ") })
    ]));
  }

  /* ── the forest: fetched as SVG, stripped of anything executable, then made clickable ── */
  function sanitizeSvg(text) {
    var parsed = new DOMParser().parseFromString(text, "image/svg+xml");
    var root = parsed.documentElement;
    if (!root || String(root.nodeName).toLowerCase() !== "svg") { return null; }
    if (parsed.getElementsByTagName("parsererror").length) { return null; }
    var walker = parsed.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null);
    var doomed = [];
    var node = root;
    while (node) {
      var name = String(node.nodeName).toLowerCase();
      if (name === "script" || name === "foreignobject" || name === "iframe" || name === "use"
          || name === "a" || name === "animate" || name === "set" || name === "handler") {
        doomed.push(node);
      } else {
        Array.prototype.slice.call(node.attributes || []).forEach(function (attribute) {
          var attributeName = attribute.name.toLowerCase();
          var isLink = attributeName === "href" || attributeName === "xlink:href";
          if (attributeName.indexOf("on") === 0
              || (isLink && attribute.value.trim().slice(0, 5).toLowerCase() !== "data:")) {
            node.removeAttribute(attribute.name);
          }
        });
      }
      node = walker.nextNode();
    }
    doomed.forEach(function (bad) { if (bad.parentNode) { bad.parentNode.removeChild(bad); } });
    return document.importNode(root, true);
  }

  function rowKeys(row) {
    var author = String(row.first_author || "").trim();
    var year = row.year ? String(row.year) : "";
    var keys = [];
    if (author && year) { keys.push(author + " " + year); }
    if (author) { keys.push(author); }
    if (row.label) { keys.push(String(row.label)); }
    if (row.dataset_id) { keys.push(String(row.dataset_id)); }
    return keys;
  }

  function rowSummary(row) {
    return [row.label || row.dataset_id,
      "d = " + num(row.es) + " [" + num(row.ci_low, 2) + ", " + num(row.ci_high, 2) + "]",
      "n = " + (row.n_a || "?") + "/" + (row.n_b || "?"),
      "route " + (row.route || "?"),
      row.confidence,
      row.overridden ? "human override" : ""].filter(Boolean).join(" · ");
  }

  function makeClickable(svg, rows) {
    var box = svg.getBoundingClientRect();
    var view = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width
      ? svg.viewBox.baseVal : { x: 0, y: 0, width: box.width, height: box.height };
    if (!box.height || !box.width) { return 0; }
    var scaleX = view.width / box.width, scaleY = view.height / box.height;
    var texts = Array.prototype.slice.call(svg.querySelectorAll("text"));
    var used = {}, hits = 0;
    rows.forEach(function (row) {
      var keys = rowKeys(row);
      var match = null;
      texts.some(function (node) {
        var content = String(node.textContent || "").trim();
        if (!content || used[content + "@" + node.getAttribute("y")]) { return false; }
        if (keys.indexOf(content) >= 0) { match = node; return true; }
        return false;
      });
      if (!match) { return; }
      used[String(match.textContent).trim() + "@" + match.getAttribute("y")] = true;
      var textBox = match.getBoundingClientRect();
      if (!textBox.height) { return; }
      var rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("class", "forest-hit");
      rect.setAttribute("x", String(view.x));
      rect.setAttribute("y", String(view.y + (textBox.top - box.top) * scaleY - 2));
      rect.setAttribute("width", String(view.width));
      rect.setAttribute("height", String(textBox.height * scaleY + 4));
      rect.setAttribute("tabindex", "0");
      rect.setAttribute("role", "button");
      rect.setAttribute("aria-label", "Evidence for " + rowSummary(row));
      var title = document.createElementNS(SVG_NS, "title");
      title.textContent = rowSummary(row);
      rect.appendChild(title);
      rect.addEventListener("click", function () { openDrawer(row.dataset_id, state.outcome); });
      rect.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openDrawer(row.dataset_id, state.outcome);
        }
      });
      svg.appendChild(rect);
      hits += 1;
    });
    return hits;
  }

  function renderForest(results) {
    var holder = $("forest");
    clear(holder);
    var url = (results.forest || {}).svg;
    if (!url) {
      holder.appendChild(h("p", { cls: "hint",
        text: "No forest plot: fewer than two rows reached the primary analysis." }));
      return Promise.resolve(null);
    }
    return fetch(withToken(url)).then(function (response) { return response.text(); })
      .then(function (text) {
        var svg = sanitizeSvg(text);
        if (!svg) {
          holder.appendChild(h("img", { attrs: { src: withToken((results.forest || {}).png || url),
            alt: "Forest plot" } }));
          return;
        }
        holder.appendChild(svg);
        var hits = makeClickable(svg, results.rows || []);
        if (!hits) {
          holder.appendChild(h("p", { cls: "hint",
            text: "Click a row in the extraction table for its evidence." }));
        }
      })
      .catch(function (error) { toast("The forest plot could not be drawn: " + error.message); });
  }

  /* ── extraction table ── */
  // a sha is how Canopy identifies a row; a name and a page number are how a reader checks one
  var COLUMNS = [
    ["study_label", "study"], ["dataset_label", "dataset"], ["pages", "page"],
    ["n_a", "n A"], ["n_b", "n B"], ["mean_a", "mean A"], ["mean_b", "mean B"],
    ["es", "effect"], ["ci_low", "CI low"], ["ci_high", "CI high"],
    ["route", "route"], ["confidence", "confidence"]
  ];

  function fillFilter(select, values, all) {
    clear(select);
    select.appendChild(h("option", { text: all, attrs: { value: "" } }));
    values.forEach(function (value) {
      select.appendChild(h("option", { text: value, attrs: { value: value } }));
    });
  }

  function renderTable(results) {
    var rows = results.rows || [];
    fillFilter($("filter-route"), unique(rows.map(function (r) { return r.route; })), "any route");
    fillFilter($("filter-confidence"), unique(rows.map(function (r) { return r.confidence; })),
      "any confidence");
    drawRows();
  }

  function unique(values) {
    var seen = {};
    return values.filter(function (value) {
      if (!value || seen[value]) { return false; }
      seen[value] = true;
      return true;
    }).sort();
  }

  function drawRows() {
    var results = state.results || {};
    var table = $("extraction");
    var head = table.tHead, body = table.tBodies[0];
    clear(head); clear(body);
    var headRow = h("tr");
    COLUMNS.forEach(function (column) {
      headRow.appendChild(h("th", { text: column[1], attrs: { scope: "col" } }));
    });
    headRow.appendChild(h("th", { text: "evidence", attrs: { scope: "col" } }));
    head.appendChild(headRow);

    var route = $("filter-route").value, confidence = $("filter-confidence").value;
    var needle = $("filter-text").value.trim().toLowerCase();
    (results.rows || []).forEach(function (row) {
      if (route && row.route !== route) { return; }
      if (confidence && row.confidence !== confidence) { return; }
      if (needle && JSON.stringify(row).toLowerCase().indexOf(needle) < 0) { return; }
      var tr = h("tr", { cls: "is-clickable" + (row.in_primary ? "" : " is-held"),
        attrs: { title: row.dataset_id + (row.paper_filename ? " · " + row.paper_filename : "") },
        on: { click: function () { openDrawer(row.dataset_id, state.outcome); } } });
      COLUMNS.forEach(function (column) {
        var value = row[column[0]];
        var isNumber = typeof value === "number";
        var text = isNumber ? num(value, column[0] === "n_a" || column[0] === "n_b" ? 0 : 3)
          : (value === null || value === undefined || value === "" ? "—" : String(value));
        if (column[0] === "study_label" && text === "—") { text = row.dataset_id; }
        tr.appendChild(h("td", { cls: isNumber ? "num" : "", text: text }));
      });
      var marks = h("td");
      if (row.overridden) { marks.appendChild(h("span", { cls: "badge warn", text: "△ override" })); }
      if (!row.in_primary) { marks.appendChild(h("span", { cls: "badge", text: "held" })); }
      marks.appendChild(h("span", { cls: "badge", text: "open" }));
      tr.appendChild(marks);
      body.appendChild(tr);
    });
  }

  ["filter-route", "filter-confidence"].forEach(function (id) {
    $(id).addEventListener("change", drawRows);
  });
  $("filter-text").addEventListener("input", drawRows);

  /* ── flags / review queue ── */
  function decide(entry, payload, done) {
    api("/api/runs/" + state.runId + "/overrides", { method: "POST", json: payload })
      .then(function (body) {
        toast("recorded as override #" + body.override.seq + " — re-pool to see it");
        highlightRepool(true);
        if (done) { done(); }
      })
      .catch(function (error) { toast(error.message); });
  }

  function highlightRepool(on) {
    var button = $("repool-btn");
    if (!button) { return; }
    show(button, true);
    button.classList.toggle("primary", !!on);
    button.classList.toggle("pending", !!on);
  }

  function flagTitle(entry) {
    var name = entry.study_label || entry.first_author || "";
    var parts = [name, entry.dataset_label, entry.group ? "group " + entry.group : "",
                 entry.pages].filter(Boolean);
    return parts.join(" · ") || entry.dataset_id;
  }

  function renderFlags(results) {
    var holder = $("flags");
    clear(holder);
    var queue = results.review || [];
    if (!queue.length) {
      holder.appendChild(h("p", { cls: "hint", text: "Nothing is waiting for a human." }));
    }
    queue.forEach(function (entry) {
      var impact = entry.impact_abs_delta_pooled;
      var card = h("div", { cls: "flag" }, [
        h("div", { cls: "flag-head" }, [
          h("span", { cls: "id", text: flagTitle(entry),
            attrs: { title: entry.dataset_id + " · " + entry.outcome_key } }),
          h("span", { cls: "flag-impact", text: impact === null || impact === undefined
            ? "impact unknown" : "|Δ pooled| " + num(impact) })
        ]),
        h("p", { text: entry.reason || "held for review" })
      ]);
      (entry.candidates || []).forEach(function (candidate) {
        card.appendChild(h("p", { cls: "hint", text:
          candidate.model + " read " + num(candidate.value) + " ± " + num(candidate.dispersion_value)
          + " (" + (candidate.dispersion_type || "?") + "), n=" + (candidate.n || "?")
          + ", p." + (candidate.page || "?") }));
      });

      var outcomeKey = entry.outcome_key || state.outcome;
      var reason = h("input", { attrs: { type: "text", placeholder:
        "why — e.g. “checked against Table 2, the reading is right”" } });
      var accept = h("button", {
        cls: "btn primary small", text: "Accept as read", attrs: { type: "button" },
        on: { click: function () {
          var why = reason.value.trim() || "checked against the paper; the reading is right";
          accept.disabled = true;
          decide(entry, { kind: "mark_reviewed", dataset_id: entry.dataset_id,
                          outcome_key: outcomeKey, justification: why },
                 function () { card.classList.add("is-decided"); });
        } } });
      var override = h("button", {
        cls: "btn small", text: "Override value…", attrs: { type: "button" },
        on: { click: function () { openDrawer(entry.dataset_id, outcomeKey); } } });
      var exclude = h("button", {
        cls: "btn small", text: "Exclude dataset", attrs: { type: "button" },
        on: { click: function () {
          var why = reason.value.trim();
          if (!why) { toast("say why this dataset should not be pooled"); reason.focus(); return; }
          exclude.disabled = true;
          decide(entry, { kind: "exclude_dataset", dataset_id: entry.dataset_id,
                          outcome_key: outcomeKey, justification: why },
                 function () { card.classList.add("is-decided"); });
        } } });

      card.appendChild(h("label", { cls: "field" },
        [h("span", { cls: "label", text: "your note" }), reason]));
      card.appendChild(h("div", { cls: "flag-actions" }, [
        accept, override, exclude,
        h("button", { cls: "btn ghost small", text: "Open the evidence",
          attrs: { type: "button" },
          on: { click: function () { openDrawer(entry.dataset_id, outcomeKey); } } })
      ]));
      holder.appendChild(card);
    });

    (results.excluded || []).forEach(function (entry) {
      holder.appendChild(h("div", { cls: "flag" }, [
        h("div", { cls: "flag-head" }, [
          h("span", { cls: "id", text: entry.dataset_id }),
          h("span", { cls: "badge warn", text: "excluded by a reviewer" })
        ]),
        h("p", { text: entry.detail || "" })
      ]));
    });
  }

  function renderFigures(results) {
    var holder = $("figures");
    clear(holder);
    var outputs = (state.run || {}).outputs || {};
    var wanted = [["prisma.png", "PRISMA flow"], ["methods_fig.png", "Where the numbers came from"]];
    Object.keys(results.figures || {}).forEach(function (name) {
      holder.appendChild(h("figure", {}, [
        h("img", { attrs: { src: withToken(results.figures[name]), alt: name + " figure",
          loading: "lazy" } }),
        h("figcaption", { text: name })
      ]));
    });
    wanted.forEach(function (pair) {
      var relative = outputs[pair[0]];
      if (!relative) { return; }
      var url = "/api/runs/" + state.runId + "/files/" + encodeURI(relative);
      holder.appendChild(h("figure", {}, [
        h("img", { attrs: { src: withToken(url), alt: pair[1], loading: "lazy" } }),
        h("figcaption", { text: pair[1] })
      ]));
    });
    if (!holder.children.length) {
      holder.appendChild(h("p", { cls: "hint", text: "This run wrote no figures." }));
    }
  }

  function renderDownloads(results) {
    var holder = $("downloads");
    clear(holder);
    var outputs = (state.run || {}).outputs || {};
    Object.keys(outputs).sort().forEach(function (name) {
      var url = "/api/runs/" + state.runId + "/files/" + encodeURI(outputs[name]);
      holder.appendChild(h("li", {}, [
        h("a", { text: name, attrs: { href: withToken(url) } })
      ]));
    });
    if (!holder.children.length) {
      holder.appendChild(h("li", { text: "No artefacts yet." }));
    }
  }

  /* ───────────────────────────────────────────────────────── evidence drawer */
  function kv(pairs) {
    var list = h("dl", { cls: "ev-kv" });
    pairs.forEach(function (pair) {
      if (pair[1] === null || pair[1] === undefined || pair[1] === "") { return; }
      list.appendChild(h("dt", { text: pair[0] }));
      list.appendChild(h("dd", { text: String(pair[1]) }));
    });
    return list;
  }

  function section(title, kids) {
    return h("section", { cls: "ev-section" }, [h("h3", { text: title })].concat(kids || []));
  }

  function closeDrawer() {
    var drawer = $("drawer");
    if (drawer.hidden) { return; }
    show(drawer, false);
    show($("scrim"), false);
    if (state.returnFocus && document.contains(state.returnFocus)) {
      state.returnFocus.focus();                 // back where the reader was, not at the top
    }
    state.returnFocus = null;
  }
  $("drawer-close").addEventListener("click", closeDrawer);
  $("scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { closeDrawer(); }
  });

  function openDrawer(datasetId, outcomeKey) {
    if (!datasetId) { return; }
    state.returnFocus = document.activeElement;
    show($("drawer"), true);
    $("drawer-close").focus();                   // the drawer is where the keyboard is now
    show($("scrim"), window.matchMedia("(max-width: 52rem)").matches);
    var body = $("drawer-body");
    clear(body);
    body.appendChild(h("p", { cls: "hint", text: "Reading the evidence…" }));
    api("/api/runs/" + state.runId + "/evidence/" + encodeURIComponent(datasetId) + "/"
      + encodeURIComponent(outcomeKey))
      .then(function (evidence) { drawEvidence(evidence); })
      .catch(function (error) {
        clear(body);
        body.appendChild(h("p", { cls: "error", text: error.message }));
      });
  }

  function drawEvidence(evidence) {
    var body = $("drawer-body");
    clear(body);
    var citation = evidence.citation || {};
    var record = evidence.record || {};
    $("drawer-title").textContent = (citation.first_author || citation.authors || "This paper")
      + (citation.year ? " " + citation.year : "");

    body.appendChild(section("What was pooled", [
      kv([
        ["dataset", evidence.dataset_id],
        ["outcome", evidence.outcome_key],
        ["effect", num(record.es) + " [" + num(record.ci_low, 2) + ", " + num(record.ci_high, 2) + "]"],
        ["n (A/B)", (record.n_a || "?") + " / " + (record.n_b || "?")],
        ["route", record.route],
        ["confidence", record.confidence],
        ["title", citation.title]
      ])
    ]));

    (evidence.images || []).forEach(function (image) {
      body.appendChild(section("Where it was read · group " + (image.group || "?"), [
        h("img", { cls: "ev-img", attrs: { src: withToken(image.url), loading: "lazy",
          alt: "Page " + (image.page || "?") + " of the paper, with the quoted value highlighted" } }),
        image.quote ? h("blockquote", { cls: "ev-quote", text: image.quote }) : null,
        h("p", { cls: "hint", text: (image.matched ? "" : "the quote could not be located on the page — ")
          + (image.note || "") })
      ]));
    });

    var candidates = h("div", { cls: "table-wrap" });
    var table = h("table", { cls: "grid" });
    var head = h("thead", {}, [h("tr", {}, ["group", "value", "n", "route", "model", "page", "status"]
      .map(function (name) { return h("th", { text: name, attrs: { scope: "col" } }); }))]);
    var tbody = h("tbody");
    (evidence.candidates || []).forEach(function (candidate) {
      tbody.appendChild(h("tr", {}, [
        h("td", { text: candidate.group || "—" }),
        h("td", { cls: "num", text: num(candidate.mean) + " ± " + num(candidate.dispersion_value)
          + " " + (candidate.dispersion_type || "") }),
        h("td", { cls: "num", text: candidate.n === null || candidate.n === undefined ? "—" : String(candidate.n) }),
        h("td", { text: candidate.route || "—" }),
        h("td", { text: candidate.model || "—" }),
        h("td", { cls: "num", text: candidate.page === null || candidate.page === undefined ? "—" : String(candidate.page) }),
        h("td", { text: candidate.status + (candidate.grounded === false ? " · ungrounded" : "") })
      ]));
    });
    table.appendChild(head);
    table.appendChild(tbody);
    candidates.appendChild(table);
    body.appendChild(section("Every reading", [candidates]));

    (evidence.verdicts || []).forEach(function (verdict) {
      var badge = verdict.confidence === "auto_accept" ? "ok"
        : (verdict.confidence === "needs_human" ? "stop" : "warn");
      body.appendChild(section("Verification · group " + (verdict.group || "?"), [
        h("p", {}, [h("span", { cls: "badge " + badge, text: verdict.confidence }),
          document.createTextNode(" "),
          h("span", { cls: "badge", text: "vote: " + (verdict.agreement || "—") }),
          document.createTextNode(" "),
          h("span", { cls: "badge", text: "verifier: " + verdict.verifier.verdict })]),
        verdict.verifier.reason ? h("p", { text: verdict.verifier.reason }) : null,
        verdict.verifier.better_source
          ? h("p", { text: "a better source exists: " + verdict.verifier.better_source }) : null,
        verdict.adjudication_rationale
          ? h("p", { text: "adjudicator: " + verdict.adjudication_rationale }) : null,
        verdict.override_justification
          ? h("p", { text: "human override: " + verdict.override_justification }) : null,
        (verdict.confidence_reasons || []).length
          ? h("p", { cls: "hint", text: verdict.confidence_reasons.join("; ") }) : null,
        verdict.orientation_evidence
          ? h("p", { cls: "hint", text: "direction: " + verdict.orientation_evidence }) : null
      ]));
    });

    var chain = h("ol", { cls: "chain" });
    (evidence.conversion_steps || []).forEach(function (step) {
      chain.appendChild(h("li", { text: step }));
    });
    body.appendChild(section("How the effect size was computed", [
      h("p", { cls: "mono", text: evidence.conversion_chain || "—" }),
      (evidence.conversion_steps || []).length ? chain : null,
      Object.keys(evidence.routes_rejected || {}).length
        ? kv(Object.keys(evidence.routes_rejected).map(function (name) {
          return ["rejected · " + name, evidence.routes_rejected[name]];
        })) : null
    ]));

    body.appendChild(overrideForm(evidence));
  }

  /* ── the review form: one decision, one justification, appended for ever ── */
  function overrideForm(evidence) {
    var kinds = [
      ["value", "Replace a value"],
      ["mark_reviewed", "Mark as reviewed"],
      ["exclude_dataset", "Exclude this dataset"],
      ["re_extract", "Ask for a re-extraction"],
      ["eligibility", "Change the paper's eligibility"]
    ];
    var kind = h("select");
    kinds.forEach(function (pair) {
      kind.appendChild(h("option", { text: pair[1], attrs: { value: pair[0] } }));
    });
    var group = h("select");
    ["A", "B"].forEach(function (key) {
      group.appendChild(h("option", { text: "group " + key, attrs: { value: key } }));
    });
    var mean = h("input", { attrs: { type: "text", placeholder: "mean" } });
    var dispersion = h("input", { attrs: { type: "text", placeholder: "SD / SE value" } });
    var dispersionType = h("select");
    ["SD", "SE", "CI95", "IQR", "RANGE", "UNKNOWN"].forEach(function (name) {
      dispersionType.appendChild(h("option", { text: name, attrs: { value: name } }));
    });
    var n = h("input", { attrs: { type: "number", min: "1", placeholder: "n" } });
    var hint = h("input", { attrs: { type: "text", placeholder: "what should be read instead" } });
    var eligible = h("select");
    [["true", "eligible"], ["false", "not eligible"]].forEach(function (pair) {
      eligible.appendChild(h("option", { text: pair[1], attrs: { value: pair[0] } }));
    });
    var justification = h("textarea", { attrs: { rows: 2,
      placeholder: "why — this is the record a reader of your review will check" } });

    var valueRow = h("div", { cls: "grid-3" }, [
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "group" }), group]),
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "mean" }), mean]),
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "dispersion" }), dispersion]),
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "type" }), dispersionType]),
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "n" }), n])
    ]);
    var hintRow = h("label", { cls: "field" }, [h("span", { cls: "label", text: "hint" }), hint]);
    var eligibleRow = h("label", { cls: "field" },
      [h("span", { cls: "label", text: "this paper is" }), eligible]);
    hintRow.hidden = true;
    eligibleRow.hidden = true;

    kind.addEventListener("change", function () {
      valueRow.hidden = kind.value !== "value";
      hintRow.hidden = kind.value !== "re_extract";
      eligibleRow.hidden = kind.value !== "eligibility";
    });

    var status = h("p", { cls: "hint", text: "" });
    var submit = h("button", { cls: "btn primary", text: "Record the decision",
      attrs: { type: "button" }, on: { click: function () {
        var payload = {
          kind: kind.value, dataset_id: evidence.dataset_id, outcome_key: evidence.outcome_key,
          paper_id: evidence.paper_id, justification: justification.value.trim()
        };
        if (kind.value === "value") {
          payload.group = group.value;
          if (mean.value) { payload.mean = Number(mean.value); }
          if (dispersion.value) { payload.dispersion_value = Number(dispersion.value); }
          payload.dispersion_type = dispersionType.value;
          if (n.value) { payload.n = Number(n.value); }
        }
        if (kind.value === "re_extract") { payload.hint = hint.value.trim(); }
        if (kind.value === "eligibility") { payload.eligible = eligible.value === "true"; }
        submit.disabled = true;
        api("/api/runs/" + state.runId + "/overrides", { method: "POST", json: payload })
          .then(function (body) {
            status.textContent = "recorded as override #" + body.override.seq
              + " — re-pool to see it in the plot";
            justification.value = "";
            highlightRepool(true);
          })
          .catch(function (error) { status.textContent = error.message; })
          .then(function () { submit.disabled = false; });
      } } });

    var repool = h("button", { cls: "btn", text: "Re-pool with the overrides",
      attrs: { type: "button" }, on: { click: function () {
        repool.disabled = true;
        api("/api/runs/" + state.runId + "/repool", { method: "POST" })
          .then(function (summary) {
            toast(summary.applied + " override(s) applied"
              + (summary.pending.length ? ", " + summary.pending.length + " need a re-run" : ""));
            return refreshRun().then(loadResults);
          })
          .catch(function (error) { toast(error.message); })
          .then(function () { repool.disabled = false; });
      } } });

    var existing = h("div");
    (evidence.overrides || []).forEach(function (override) {
      existing.appendChild(h("p", { cls: "hint", text: "#" + override.seq + " " + override.kind
        + " · " + override.at + " · " + override.justification }));
    });

    return section("Your decision", [
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "what to change" }), kind]),
      valueRow, hintRow, eligibleRow,
      h("label", { cls: "field" },
        [h("span", { cls: "label", text: "justification" }), justification]),
      h("div", { cls: "actions" }, [repool, submit]),
      status, existing
    ]);
  }

  /* ───────────────────────────────────────────────────────── runs list */
  function loadRuns() {
    api("/api/runs").then(function (body) {
      var list = $("run-list");
      clear(list);
      (body.runs || []).forEach(function (run) {
        var open = h("button", { cls: "btn small", text: "Open", attrs: { type: "button" },
          on: { click: function () {
            state.runId = run.run_id;
            state.token = run.token || window.prompt("This run's token:") || "";
            state.started = run.status !== "created";
            state.outcome = "";
            state.papers = {};
            clear($("stage-grid").tBodies[0]);
            refreshRun().then(loadResults).catch(function (error) { toast(error.message); });
          } } });
        list.appendChild(h("li", {}, [
          h("span", {}, [
            h("strong", { text: run.title || run.run_id }),
            h("span", { cls: "hint", text: " " + run.run_id })
          ]),
          h("span", {}, [
            h("span", { cls: "badge " + (run.status === "done" ? "ok"
              : (run.status === "error" ? "stop" : "")), text: run.status }),
            document.createTextNode(" " + money(run.cost_usd) + " "),
            open
          ])
        ]));
      });
      if (!list.children.length) {
        list.appendChild(h("li", { text: "No runs yet." }));
      }
    }).catch(function (error) { toast(error.message); });
  }

  /* ───────────────────────────────────────────────────────── boot */
  addOutcome(null);
  loadSettings().catch(function (error) { toast(error.message); });
  loadExamples().catch(function (error) { toast(error.message); });

  // a reload must not lose the run: re-attach to the one this browser was watching, if it is
  // still there and its token still works
  var saved = stored(STORE_CURRENT, "");
  if (saved) {
    var savedToken = tokenFor(saved);
    if (savedToken) {
      attach(saved, savedToken).catch(function () { forgetCurrentRun(); });
    }
  }
})();
