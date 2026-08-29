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

  // …and a hidden pane must stop demanding its fields. A `required` control inside a pane the
  // reviewer has switched away from still fails the form's own validation, and the browser then
  // refuses to submit and tries to focus a control it cannot see ("An invalid form control with
  // name='' is not focusable"). The Run button is the form's submit, so YAML mode could not start
  // a run at all — silently, with the reason only in the console — while Dry run, a plain button,
  // worked. Written against the pane rather than against `p-title` so a `required` added to either
  // pane later cannot bring the failure back.
  function demandFields(pane, on) {
    if (!pane) { return; }
    var fields = pane.querySelectorAll("[required], [data-was-required]");
    Array.prototype.forEach.call(fields, function (field) {
      if (on) {
        if (field.hasAttribute("data-was-required")) {
          field.required = true;
          field.removeAttribute("data-was-required");
        }
      } else if (field.required) {
        field.required = false;
        field.setAttribute("data-was-required", "1");
      }
    });
  }

  function showPane(pane, on) { show(pane, on); demandFields(pane, on); }

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
  // real plurals, everywhere a count is written: "1 paper", "3 papers" — never "paper(s)"
  function plural(count, word) {
    var n = Number(count) || 0;
    return n + " " + word + (n === 1 ? "" : "s");
  }

  /* ───────────────────────────────────────────────────────── state + API */
  var state = {
    settings: null, examples: [], runId: "", token: "", run: null, results: null,
    outcome: "", files: [], events: [], papers: {}, source: null, mode: "guided",
    started: false, returnFocus: null, live: {}, poll: null, names: {},
    // which analysis line the headline, the forest and the row badges are about. "strict" is
    // the primary analysis; "best_guess" is the second line and the one the results page lands
    // on when the run has it (see `storedLine`). The two are never on screen at the same time
    // (DECISION A).
    line: "strict",

    // ── the paper search. Deliberately none of these is called `mode` (that is the protocol
    // editor's) or `source` (that is the run's EventSource): two screens sharing one word is how
    // a toggle on one of them starts hiding a pane on the other.
    papersFrom: "upload",
    searchId: "", searchToken: "", search: null, searchSource: null, searchPoll: null,
    searchEvents: [], announced: 0,
    // a keep/drop whose POST is still in flight. The 5-second poll is authoritative about
    // everything EXCEPT these, or a tick would visibly flip back while the server agrees with it.
    pendingDecisions: {}
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

  /* Which line this TAB is looking at, per run. Deliberately sessionStorage rather than the
     local one the tokens live in: "show me the guess" is a thing a reviewer does while reading,
     not a preference — a link they send, or the same run opened tomorrow, starts on the primary
     analysis again. */
  function lineKey(runId) { return "canopy.line." + (runId || state.runId); }

  function storedLine(runId) {
    // the page LANDS on the best-guess line when the run has one (the fuller picture, labelled
    // as not the primary analysis); a reviewer who switched to strict in this tab stays there.
    // `drawLineToggle` falls back to strict on a run whose guess admitted nothing.
    try {
      var stored = window.sessionStorage.getItem(lineKey(runId));
      return stored === "strict" ? "strict" : "best_guess";
    } catch (err) { return "best_guess"; }
  }

  function rememberLine(runId, line) {
    try { window.sessionStorage.setItem(lineKey(runId), line); }
    catch (err) { /* private mode: the toggle still works, it just does not survive a reload */ }
  }

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
  var SCREENS = ["new", "search", "monitor", "results", "runs"];

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
      pill.textContent = settings.api_key_configured ? "key loaded" : "no key";
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

      // the search's two caps are pre-filled with the numbers this server will actually hold it
      // to, so the form shows the truth rather than a hopeful default of its own
      $("f-max-usd").value = settings.search_max_usd;
      $("f-max-screened").value = settings.search_max_screened;
      // A search with no model still runs AND still fetches — template queries, nothing screened,
      // every open-access copy on disk — so this is a warning about what the list will and will
      // not have been sorted by, never a locked door.
      var keyNote = $("find-key-note");
      keyNote.textContent = "Sorting the papers for you needs a model. Without one the search "
        + "still runs and still fetches the open-access PDFs: the queries come from your own "
        + "words and no abstract is read, so every paper is listed for you to judge and none is "
        + "ticked. Canopy reads ANTHROPIC_API_KEY from .env; nothing on this page ever shows it.";
      show(keyNote, !!settings.search_key_required);
    });
  }

  function loadExamples() {
    return api("/api/protocols/examples").then(function (body) {
      state.examples = body.examples || [];
      var picker = $("example-picker");
      clear(picker);
      picker.appendChild(h("option", { text: "Blank protocol", attrs: { value: "" } }));
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
      showPane($("guided"), state.mode === "guided");
      showPane($("yaml-mode"), state.mode === "yaml");
    });
  });

  $("add-outcome").addEventListener("click", function () { addOutcome(null); });

  $("draft-btn").addEventListener("click", function () {
    var sentence = $("draft-sentence").value.trim();
    if (!sentence) { toast("Describe your review in a sentence first."); return; }
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
        toast("Draft ready. Check every line before you run.");
      })
      .catch(function (error) { toast(error.message); })
      .then(function () { button.disabled = false; button.textContent = "Draft a protocol"; });
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
    $("files-note").textContent = plural(chosen.length, "PDF") + " ready"
      + (skipped > 0 ? " · " + plural(skipped, "non-PDF file") + " ignored" : "");
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
    note.textContent = message || ("Uploading " + plural(state.files.length, "file")
      + "… " + percent + "%");
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
        button.textContent = "Dry run on 3 papers";
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
            reject(new Error("The dry run is taking too long. Watch the log instead."));
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
    $("stage-summary").textContent = plural(Object.keys(state.papers).length, "paper");
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
    $("log-count").textContent = plural(state.events.length, "event");
    $("stage-summary").textContent = plural(Object.keys(state.papers).length, "paper");
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
        : "Run " + (event.status || "finished") + (event.message ? ". " + event.message : ""));
      refreshRun().then(function () {
        if (event.status === "done") { loadResults(); }
      }).catch(function (error) { toast(error.message); });
    });
    source.onerror = function () { source.close(); state.source = null; };
  }

  $("cancel-btn").addEventListener("click", function () {
    if (!state.runId) {          // the create request is still uploading: there is nothing to cancel
      toast("Papers are still uploading, so there is no run to stop yet. "
            + "Reload the page to abandon the upload.");
      return;
    }
    api("/api/runs/" + state.runId + "/cancel", { method: "POST" })
      .then(function (body) { toast("Run " + body.status + ". Finished work is saved."); })
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
        + plural((run.papers || []).length, "paper") + "." }));
      if (run.status !== "error") {
        banner.appendChild(h("button", {
          cls: "btn ghost small", text: "Open results", attrs: { type: "button" },
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
      ["forest", "table", "questions", "flags", "figures", "downloads"].forEach(function (key) {
        show($("pane-" + key), key === name);
      });
      if (name === "questions") { loadQuestions(); }
    });
  });

  /* ── questions: every held cell as one thing to decide, with its screenshot ── */
  function loadQuestions() {
    if (!state.runId) { return Promise.resolve(null); }
    return api("/api/runs/" + state.runId + "/questions")
      .then(function (body) {
        renderQuestions(body.questions || [], body.n_open || 0, body.n_pending || 0);
      })
      .catch(function (error) {
        clear($("questions"));
        $("questions").appendChild(h("p", { cls: "hint", text: error.message }));
      });
  }

  function renderQuestions(questions, nOpen, nPending) {
    var badge = $("questions-count");
    badge.textContent = String(nOpen);
    badge.title = nPending
      ? plural(nPending, "answered decision") + " waiting for a re-run" : "";
    show(badge, nOpen > 0 || nPending > 0);
    var holder = $("questions");
    clear(holder);
    if (!questions.length) {
      holder.appendChild(h("p", { cls: "hint",
        text: "No open questions. Every value was settled during verification." }));
      return;
    }
    // DECISION C4: what answering is WORTH decides where a card sits. The tool never answers a
    // low-impact card for the reviewer — it just stops putting it at the top of their day.
    var moving = [], low = [], answered = [], settled = [];
    questions.forEach(function (q) {
      // a choice the TOOL made for itself is its own bucket. It is `answered` — so it never sits
      // above a card that still blocks — but "Answered" means a person decided, and filing it
      // there said one had.
      if (q.status === "settled") { settled.push(q); }
      else if (q.answered) { answered.push(q); }
      else if (q.impact_band === "low") { low.push(q); }
      else { moving.push(q); }
    });
    [["Decisions that move the result", moving,
      "Ordered by how far the pooled estimate moves when this is settled."],
     ["Low impact. Answer if you have time", low,
      "Every answer on offer here moves this row's effect size by less than 0.10 and the pooled "
      + "estimate by less than 0.05. They are still open; nothing was decided for you."],
     ["Answered", answered,
      "Kept on the page with what was decided and when. A re-pool has already used them."],
     ["Resolved automatically. Confirm or change", settled,
      "Nothing here blocked the run. Where a paper reports one outcome two ways, the map picked "
      + "one and the numbers above were read with it. Each card shows the reading it set aside "
      + "and the words it decided on; changing one re-reads that cell on the next resume."]
    ].forEach(function (part) {
      if (!part[1].length) { return; }
      var section = h("section", { cls: "q-section" }, [
        h("h3", { text: part[0] + " (" + part[1].length + ")" }),
        h("p", { cls: "hint", text: part[2] })
      ]);
      part[1].forEach(function (q) { section.appendChild(questionCard(q)); });
      holder.appendChild(section);
    });
  }

  // a question whose answers are a menu, not a number: a direction, an inclusion, a choice of
  // measure, a paper's eligibility, which pair of numbers a row is and how many people an arm
  // analysed are all answered by picking one of the things offered, so a "type it yourself" box
  // could only add a way to answer with nothing. (The analysed-n card's own "type it" option is
  // an OPTION, with the two boxes it needs — that is a different thing from free text.)
  function isChoiceOnly(q) {
    return q.kind === "orientation" || q.kind === "include_dataset" || q.kind === "which_measure"
      || q.kind === "include_paper" || q.kind === "precedence_override" || q.kind === "analysed_n";
  }

  // what a card is worth, in the words the queue used to price it
  function impactPill(q) {
    var basis = q.impact_basis || "unknown";
    if (basis === "unknown") {
      return h("span", { cls: "pill", text: "Impact not estimated",
        attrs: { title: "nothing on the record says what answering this would move" } });
    }
    var moves = basis === "pooled" ? "moves the pooled estimate by "
                                   : "spreads this row's effect size over ";
    return h("span", { cls: q.impact_band === "low" ? "pill" : "pill on",
      text: (q.impact_band === "low" ? "low impact · " : "") + moves + num(q.impact, 2),
      attrs: { title: basis === "pooled"
        ? "how far the pooled estimate moves if this row's value changes"
        : "the spread of the effect sizes this card's own options imply" } });
  }

  // A card whose options are COMBINATIONS: one answer settles both groups at once, and each
  // option carries the per-slot keys (`a`, `b`) it is made of. The page therefore asks the two
  // questions the card is made of, one radio group per slot, and resolves the pair back to the
  // combination the server offered — the key alone is positional twice over.
  function isCombination(q) {
    var options = q.options || [];
    return (q.slots || []).length >= 2 && options.length > 0
      && options.every(function (o) { return o.a && o.b; });
  }

  function slotPrefix(slot, i) {
    var group = String((slot || {}).group || "").toUpperCase();
    return group === "B" ? "b" : (group === "A" ? "a" : String(i));
  }

  function combinationOf(q, form) {
    var picks = {};
    (q.slots || []).forEach(function (slot, i) {
      var prefix = slotPrefix(slot, i);
      var input = form.querySelector("input[name=option_" + prefix + "]:checked");
      if (input) { picks[prefix] = input.value; }
    });
    if (picks.a === undefined || picks.b === undefined) { return null; }
    var found = null;
    (q.options || []).forEach(function (o) {
      if (o.a === picks.a && o.b === picks.b) { found = o; }
    });
    return found;
  }

  function optionText(o) {
    return o.label
      + (o.backed_by && o.backed_by.length ? " · " + o.backed_by.join(", ") : "")
      + (o.quote ? " “" + String(o.quote).slice(0, 140) + "”" : "");
  }

  function questionCard(q) {
    var head = h("div", { cls: "q-head" }, [
      h("span", { cls: "q-num", text: "#" + q.number }),
      h("span", { cls: "q-paper", text: q.paper + (q.dataset_label ? " · " + q.dataset_label : "") }),
      h("span", { cls: "q-cell", text: (q.outcome_key || "").replace(/_/g, " ")
        + (q.group_label ? " · " + q.group_label : "") }),
      // three states, not two: a decision whose consequence has not happened yet is not
      // "answered" — the extraction it decides has to be bought before anything changes.
      (q.status === "pending_rerun")
        ? h("span", { cls: "pill off", text: "Answered. Applies on the next resume.",
                      attrs: { title: q.pending_why || "" } })
        // …and a fourth: a choice the TOOL made for itself, which blocked nothing and was never
        // asked. Not "answered" in the sense a reviewer means by it, and the pill has to say so or
        // the card reads as one somebody already looked at.
        : (q.status === "settled")
        ? h("span", { cls: "pill off", text: "Resolved automatically. Confirm or change.",
                      attrs: { title: q.why || "" } })
        : q.answered ? h("span", { cls: "pill ok", text: "answered" })
                     : h("span", { cls: "pill", text: (q.kind === "cell" && q.slot_answers)
                         ? (q.slots || []).length + " questions, one cell"
                         : q.kind.replace(/_/g, " ") }),
      impactPill(q)
    ]);
    var body = h("div", { cls: "q-body" });
    if (q.image && q.image.url) {
      // every run file is served behind the run's own token — an <img> carries no headers, so
      // the token has to be in the URL. Without it the browser gets a 401 and the reviewer is
      // asked "which series is this?" beside a broken-image icon.
      var imageUrl = withToken(q.image.url);
      var img = h("img", { cls: "q-image", attrs: { src: imageUrl, alt: q.where || "evidence",
        loading: "lazy" } });
      body.appendChild(h("a", { attrs: { href: imageUrl, target: "_blank", rel: "noopener" } },
        [img]));
    }
    var right = h("div", { cls: "q-right" });
    right.appendChild(h("p", { cls: "q-prompt", text: q.prompt }));
    // where the ROW this card is about stands while the card is open: the best-guess line either
    // admits it under a rule or vetoes it, and which of those it is changes what answering is for
    if (q.status_line) {
      right.appendChild(h("p", { cls: "q-status", text: q.status_line }));
    }
    var form = h("form", { cls: "q-form", on: { submit: function (event) {
      event.preventDefault();
      submitAnswer(q, form);
    } } });
    var combination = isCombination(q);
    var implied = h("p", { cls: "hint" });
    if (combination) {
      (q.slots || []).forEach(function (slot, i) {
        var prefix = slotPrefix(slot, i);
        var box = h("fieldset", { cls: "q-slot" }, [
          h("legend", { text: (slot.group_label || ("group " + slot.group)) + " · "
            + String(slot.kind || "").replace(/_/g, " ") })
        ]);
        if (slot.prompt) { box.appendChild(h("p", { cls: "hint", text: slot.prompt })); }
        (slot.options || []).forEach(function (o, j) {
          box.appendChild(h("label", { cls: "q-option" }, [
            h("input", { attrs: { type: "radio", name: "option_" + prefix, value: o.key,
              id: "q" + q.number + "-" + prefix + j } }),
            h("span", { text: optionText(o) })
          ]));
        });
        // history, never a tick: what this group's recorded answers already cleared, and when
        (slot.settled || []).forEach(function (was) {
          box.appendChild(h("p", { cls: "q-was", text: "already recorded"
            + (was.at ? " (" + was.at + ")" : "") + ": " + (was.justification || "") }));
        });
        form.appendChild(box);
      });
      form.appendChild(implied);
    } else {
      (q.options || []).forEach(function (o, i) {
        var id = "q" + q.number + "-o" + i;
        var wrap = h("label", { cls: "q-option" }, [
          h("input", { attrs: { type: "radio", name: "option", value: o.key, id: id,
            "data-fingerprint": o.fingerprint || "" } }),
          h("span", { text: optionText(o) })
        ]);
        form.appendChild(wrap);
        // an option that is a FORM rather than an answer (the analysed-n card's "type it"): the
        // boxes it is waiting for, named as it names them
        (o.needs_input || []).forEach(function (field) {
          wrap.appendChild(h("input", { cls: "q-needs", attrs: { type: "number", step: "1",
            name: field, placeholder: field.replace(/_/g, " ") } }));
        });
      });
    }
    // A card whose slots are answered ONE AT A TIME (`slot_answers`): everything still open on
    // one cell, or one paper's analysed sizes, each slot with its own options and its own
    // record. Nothing is combined — a slot left blank stays open, and the card stays open with
    // it. (A `pair` is the other shape: its slots resolve to one combination above.)
    if (q.slot_answers) {
      (q.slots || []).forEach(function (slot, i) {
        // an answerable slot gets its radios; a slot already answered (a paper's analysed-n
        // dataset) is shown as history only; a slot the card only carries for context is not
        // drawn here at all (it is listed under "why the tool could not decide")
        if (!slot.answerable && !(slot.settled || []).length) { return; }
        var box = h("fieldset", { cls: "q-slot", attrs: { "data-slot": slot.member_id || "" } }, [
          h("legend", { text: (slot.group_label || ("group " + slot.group)) + " · "
            + String(slot.kind || "").replace(/_/g, " ")
            + (slot.answerable ? "" : " · answered") })
        ]);
        if (slot.prompt && slot.answerable) {
          box.appendChild(h("p", { cls: "hint", text: slot.prompt }));
        }
        (slot.answerable ? (slot.options || []) : []).forEach(function (o, j) {
          var wrap = h("label", { cls: "q-option" }, [
            h("input", { attrs: { type: "radio", name: "slot_" + i, value: o.key,
              id: "q" + q.number + "-s" + i + "-" + j, "data-fingerprint": o.fingerprint || "" } }),
            h("span", { text: optionText(o) })
          ]);
          box.appendChild(wrap);
          (o.needs_input || []).forEach(function (field) {
            wrap.appendChild(h("input", { cls: "q-needs", attrs: { type: "number", step: "1",
              name: field + "__" + i, placeholder: field.replace(/_/g, " ") } }));
          });
        });
        (slot.settled || []).forEach(function (was) {
          box.appendChild(h("p", { cls: "q-was", text: "already recorded"
            + (was.at ? " (" + was.at + ")" : "") + ": " + (was.justification || "") }));
        });
        form.appendChild(box);
      });
    }
    var freeId = "q" + q.number + "-free";
    if (!isChoiceOnly(q)) {
      // on a card answered slot by slot there is no card-level option list for this radio to
      // sit in — a radio with no siblings can never be unchecked — so there it is a checkbox
      form.appendChild(h("label", { cls: "q-option" }, [
        h("input", { attrs: q.slot_answers
          ? { type: "checkbox", name: "free_toggle", value: "__free__", id: freeId }
          : { type: "radio", name: "option", value: "__free__", id: freeId } }),
        h("span", { text: (q.kind === "no_value")
          ? "it is here (say where), or it is not reported"
          : "None of these. I will type it." })
      ]));
    }
    var free = h("div", { cls: "q-free", attrs: { hidden: true } });
    if (combination || (q.slot_answers && q.kind === "cell")) {
      // a typed value on a card that settles two cells has to say WHICH cell it is: the record
      // this becomes names one group, and the slot it names is the question it answers
      var who = h("select", { attrs: { name: "group" } });
      (q.slots || []).forEach(function (slot) {
        who.appendChild(h("option", { text: slot.group_label || ("group " + slot.group),
          attrs: { value: slot.group } }));
      });
      free.appendChild(who);
    }
    if (isChoiceOnly(q)) {
      free.appendChild(h("span", { cls: "hint", text: "" }));
    } else if (q.kind === "no_value") {
      free.appendChild(h("input", { attrs: { type: "text", name: "hint",
        placeholder: "e.g. Table 2, row 'older', p. 5. Leave empty for 'not reported'." } }));
    } else {
      free.appendChild(h("input", { attrs: { type: "number", step: "any", name: "mean",
        placeholder: "mean" + (q.unit ? " (" + q.unit + ")" : "") } }));
      free.appendChild(h("input", { attrs: { type: "number", step: "any", name: "dispersion_value",
        placeholder: "spread" } }));
      var sel = h("select", { attrs: { name: "dispersion_type" } });
      ["", "SD", "SE", "CI95", "IQR", "RANGE"].forEach(function (t) {
        sel.appendChild(h("option", { text: t || "spread type", attrs: { value: t } }));
      });
      free.appendChild(sel);
      free.appendChild(h("input", { attrs: { type: "number", step: "1", name: "n",
        placeholder: "n" } }));
    }
    form.appendChild(free);
    form.addEventListener("change", function () {
      var picked = form.querySelector("input[name=option]:checked");
      var freeBox = form.querySelector("input[name=free_toggle]");
      show(free, (!!picked && picked.value === "__free__") || !!(freeBox && freeBox.checked));
      if (combination) {
        var chosen = combinationOf(q, form);
        implied.textContent = chosen ? chosen.label
          : "Pick one answer for each group; what the pair implies appears here.";
      }
    });
    if (combination) {
      implied.textContent = "Pick one answer for each group; what the pair implies appears here.";
    }
    form.appendChild(h("input", { cls: "q-note", attrs: { type: "text", name: "note",
      placeholder: (q.kind === "orientation")
        ? "the sentence in the paper that says so, recorded with the direction"
        : isChoiceOnly(q)
        ? "why: the rule or the sentence that decides it, recorded with the answer"
        : "note for the record (optional)" } }));
    var actions = h("div", { cls: "q-actions" }, [
      h("button", { cls: "btn small", attrs: { type: "submit" }, text: "Answer & re-pool" })
    ]);
    // "exclude this cell" is offered only where an exclusion is what it would write.
    //   `which_measure`   — the dataset has nothing extracted, so the re-pool has no row to drop;
    //                       excluding it is the `include_dataset` question's own answer.
    //   `analysed_n`      — the card writes a `group_n`, and an exclude posted through it arrives
    //                       as a `group_n` with no sizes, which the log refuses (422).
    //   a measure-scoped card — its `dataset_id` is empty, so the exclusion names no dataset (422).
    //   `precedence_override` — the card HAS an exclude option of its own, with the words that
    //                       say what is being excluded and why; a second, wordless one wrote the
    //                       opposite decision for as long as the answer read only the option key.
    var inertExclude = q.kind === "which_measure" || q.kind === "analysed_n"
      || q.kind === "precedence_override" || q.scope === "measure";
    if (!inertExclude) {
      actions.appendChild(h("button", { cls: "btn small ghost", attrs: { type: "button" },
        // one decision can be about more than one cell, and the button has to say so: a card
        // that folded two groups excludes the dataset, not "this cell"
        text: (q.cells || []).length > 1 ? "Exclude these cells" : "Exclude this cell",
        on: { click: function () { submitAnswer(q, form, true); } } }));
    }
    form.appendChild(actions);
    right.appendChild(form);
    // what this one answer settles: a fold must never hide a hold, so the cells are listed
    var cells = q.cells || [];
    if (cells.length > 1 || (q.scope && q.scope !== "cell" && q.scope !== "paper")) {
      var list = h("ul");
      cells.forEach(function (cell) {
        list.appendChild(h("li", { text: (cell.dataset_label || cell.dataset_id)
          + " · " + String(cell.outcome_key || "").replace(/_/g, " ")
          + (cell.group_label || cell.group ? " · " + (cell.group_label || cell.group) : "") }));
      });
      right.appendChild(h("details", { cls: "q-settles" }, [
        h("summary", { text: "this settles " + plural(cells.length, "cell") }), list
      ]));
    }
    // the reason, one sentence per thing that is holding it — a card folded from two cells
    // carries both cells' reasons, and a wall of them joined by `||` reads as neither
    var why = h("details", { cls: "q-why" }, [
      h("summary", { text: q.status === "settled" ? "what Canopy decided, and on what"
                                                : "why Canopy could not decide" })
    ]);
    String(q.why || "").split(" || ").forEach(function (reason) {
      if (reason.trim()) { why.appendChild(h("p", { text: reason.trim() })); }
    });
    (q.slots || []).forEach(function (slot) {
      if ((q.slots || []).length < 2 || !slot.why) { return; }
      why.appendChild(h("p", { cls: "hint",
        text: (slot.group_label || ("group " + slot.group)) + " was asked "
          + String(slot.kind || "").replace(/_/g, " ") + ": " + slot.why }));
    });
    right.appendChild(why);
    if (q.answered && q.answers && q.answers.length) {
      right.appendChild(h("p", { cls: "hint",
        text: "Answered: " + (q.answers[q.answers.length - 1].justification || "")
          + (q.status === "pending_rerun" ? " · Not applied yet: " + (q.pending_why || "") : "") }));
    }
    body.appendChild(right);
    return h("section", { cls: "card q-card"
      + (q.answered && q.status !== "pending_rerun" && q.status !== "settled"
         ? " is-answered" : ""),
      attrs: { "data-question": q.number } }, [head, body]);
  }

  function submitAnswer(q, form, exclude) {
    var picked = form.querySelector("input[name=option]:checked");
    var freeBox = form.querySelector("input[name=free_toggle]");
    var free = (!!picked && picked.value === "__free__") || !!(freeBox && freeBox.checked);
    // a combination card has no flat list of radios: the answer is the pair of per-slot picks,
    // resolved back to the option the server offered (and to ITS fingerprint)
    var combined = (!free && isCombination(q)) ? combinationOf(q, form) : null;
    // the number is a position in a list that moves as answers land, so the answer names the
    // question it answers and the server refuses it if that is no longer question #n.
    var payload = { id: q.id, note: (form.querySelector("input[name=note]") || {}).value || "" };
    // the slots answered one at a time (`slot_answers`): each pick names its slot and echoes the
    // fingerprint it was shown under; a slot left blank is simply not in the list
    var slotPicks = [];
    if (q.slot_answers && !exclude) {
      (q.slots || []).forEach(function (slot, i) {
        if (!slot.answerable) { return; }
        var input = form.querySelector("input[name=slot_" + i + "]:checked");
        if (!input) { return; }
        var entry = { slot: slot.member_id || "", option: input.value,
                      option_fingerprint: input.getAttribute("data-fingerprint") || "",
                      note: payload.note };
        var chosenSlot = (slot.options || []).filter(function (o) {
          return o.key === input.value;
        })[0] || {};
        (chosenSlot.needs_input || []).forEach(function (field) {
          var box = form.querySelector("[name=" + field + "__" + i + "]");
          if (box && box.value !== "") { entry[field] = box.value; }
          else { entry.__missing = true; }
        });
        slotPicks.push(entry);
      });
      if (slotPicks.some(function (e) { return e.__missing; })) {
        toast("Fill in the numbers this answer is waiting for."); return;
      }
      slotPicks.forEach(function (e) { delete e.__missing; });
    }
    if (exclude) { payload.exclude = true; }
    else if (free && slotPicks.length) {
      // a typed value and slot picks in one submit would drop one of them on the floor: the
      // typed value goes to one group, the picks to others, and the page cannot know which
      toast("Answer the slots above, or type a value. Not both in one submit."); return;
    } else if (!picked && !combined && !slotPicks.length && !free) {
      toast(isCombination(q) ? "Choose an answer for each group first." : "Choose an answer first.");
      return;
    } else if (slotPicks.length && !picked && !combined) {
      payload.slots = slotPicks;
    } else if (free) {
      ["hint", "mean", "dispersion_value", "dispersion_type", "n", "group"].forEach(
        function (name) {
          var input = form.querySelector("[name=" + name + "]");
          if (input && input.value !== "") { payload[name] = input.value; }
        });
      if (q.kind !== "no_value" && !isChoiceOnly(q) && payload.mean === undefined
          && payload.dispersion_value === undefined
          && payload.n === undefined) { toast("Type at least a mean, a spread or an n."); return; }
      if (!payload.note && (payload.mean !== undefined || payload.dispersion_value !== undefined
          || payload.n !== undefined)) {
        toast("A typed number needs a note saying where it comes from.");
        return;
      }
    } else {
      payload.option = combined ? combined.key : picked.value;
      // what this option MEANT when it was drawn: the keys are positional and the list is rebuilt
      // on every request, so the server refuses an answer echoing a number it no longer offers.
      payload.option_fingerprint = combined
        ? (combined.fingerprint || "")
        : (picked.getAttribute("data-fingerprint") || "");
      // an option that is a form (the analysed-n card's "type it") carries the boxes it named
      var chosen = combined || (q.options || []).filter(function (o) {
        return o.key === picked.value;
      })[0] || {};
      var missing = false;
      (chosen.needs_input || []).forEach(function (field) {
        var input = form.querySelector("[name=" + field + "]");
        if (input && input.value !== "") { payload[field] = input.value; }
        else { missing = true; }
      });
      if (missing) { toast("Fill in the numbers this answer is waiting for."); return; }
    }
    // a card-level pick and slot picks travel together (a precedence decision with the
    // refutation it carries): one POST, one record per thing decided
    if (slotPicks.length && !payload.slots) { payload.slots = slotPicks; }
    var buttons = form.querySelectorAll("button");
    Array.prototype.forEach.call(buttons, function (b) { b.disabled = true; });
    api("/api/runs/" + state.runId + "/questions/" + q.number + "/answer",
        { method: "POST", json: payload })
      .then(function (body) {
        // an answer the tool cannot act on itself says so, in the reviewer's own moment: a map
        // answer waits for the resume that buys the extraction, and "recorded" alone would read
        // as "done". One decision can be more than one record (§C1) — every one of them is
        // checked, because a card that settles two cells can have one of them waiting.
        var records = body.overrides || (body.override ? [body.override] : []);
        var seqs = records.map(function (record) { return record.seq; });
        var waiting = (body.repool && body.repool.pending || []).filter(function (p) {
          return seqs.indexOf(p.seq) >= 0;
        })[0];
        var wrote = records.length > 1 ? " (" + records.length + " overrides)" : "";
        var waitingCount = body.n_pending
          ? " " + plural(body.n_pending, "answered decision") + " waiting for a re-run." : "";
        toast(waiting ? "Recorded" + wrote + ". " + waiting.why
                      : "Recorded" + wrote + ". "
                        + (body.n_open ? plural(body.n_open, "question") + " still open."
                                       : "No open questions.") + waitingCount);
        return refreshRun().then(loadResults).then(loadQuestions);
      })
      .catch(function (error) {
        toast(error.message);
        Array.prototype.forEach.call(buttons, function (b) { b.disabled = false; });
      });
  }

  $("repool-btn").addEventListener("click", function () {
    var button = $("repool-btn");
    button.disabled = true;
    api("/api/runs/" + state.runId + "/repool", { method: "POST" })
      .then(function (summary) {
        toast(plural(summary.applied, "override") + " applied"
          + (summary.pending.length ? ", " + summary.pending.length + " "
            + (summary.pending.length === 1 ? "needs" : "need") + " a re-run" : ""));
        highlightRepool(false);
        return refreshRun().then(loadResults);
      })
      .catch(function (error) { toast(error.message); })
      .then(function () { button.disabled = false; });
  });

  function loadResults() {
    if (!state.runId || !state.outcome) { return Promise.resolve(null); }
    if (state.runId) {
      api("/api/runs/" + state.runId + "/questions").then(function (body) {
        var badge = $("questions-count");
        badge.textContent = String(body.n_open || 0);
        show(badge, (body.n_open || 0) > 0);
      }).catch(function () { /* the run may not have finished; the tab says so when opened */ });
    }
    return api("/api/runs/" + state.runId + "/results/" + encodeURIComponent(state.outcome))
      .then(function (results) {
        state.results = results;
        // the line this tab was last reading for THIS run; the toggle is the only writer, so
        // reading it back here is what survives a reload and an outcome change
        state.line = storedLine(state.runId);
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

  /* ── the two analysis lines (DECISION A) ──────────────────────────────────────────────────
     One headline at a time. Which one is `state.line`; everything that carries a number the
     reader could quote — the pooled card, the forest, the row badges — reads it, so there is no
     state of this page in which the strict estimate and the guess are both on screen. */
  function hasGuess(results) {
    // `outputs.py` writes a `best_guess` block for every outcome, and at `n_added = 0` its k and
    // its estimate ARE the strict ones. Enabling the toggle on that put the same number on screen
    // twice under two names, the second time under the caveat that the line adds held rows —
    // which is exactly what `html.py` and `outputs.py` refuse to print (review finding 14).
    var guess = (results || {}).best_guess || {};
    return Number(guess.n_added) >= 1 && Number(guess.k) >= 2;
  }

  function guessing(results) {
    return state.line === "best_guess" && hasGuess(results || state.results);
  }

  function setLine(line) {
    state.line = line === "best_guess" ? "best_guess" : "strict";
    rememberLine(state.runId, state.line);
    var results = state.results;
    if (!results) { return Promise.resolve(null); }
    renderPooled(results);
    drawRows();
    return renderForest(results);
  }

  Array.prototype.forEach.call(document.querySelectorAll("#line-toggle .line-btn"),
    function (button) {
      button.addEventListener("click", function () {
        if (button.disabled) { return; }
        setLine(button.getAttribute("data-line"));
      });
    });

  function drawLineToggle(results) {
    var guess = (results || {}).best_guess || {};
    var possible = hasGuess(results);
    if (!possible && state.line === "best_guess") { state.line = "strict"; }
    Array.prototype.forEach.call(document.querySelectorAll("#line-toggle .line-btn"),
      function (button) {
        var mine = button.getAttribute("data-line");
        var on = mine === state.line;
        button.classList.toggle("is-on", on);
        button.setAttribute("aria-pressed", on ? "true" : "false");
        button.disabled = mine === "best_guess" && !possible;
      });
    $("line-note").textContent = !possible
      ? "No secondary line. No held row qualifies under a rule."
      : (state.line === "best_guess"
        ? guess.note || "Not the primary analysis."
        : "The primary analysis. The other line admits "
          + plural(guess.n_added || 0, "held row") + " under a rule.");
  }

  function renderPooled(results) {
    var guess = guessing(results);
    var pooled = guess ? (results.best_guess || {}) : (results.pooled || {});
    var outcome = results.outcome || {};
    var settings = results.settings || {};
    $("results-title").textContent = outcome.label || state.outcome;
    $("results-sub").textContent = outcome.definition || "";
    drawLineToggle(results);
    renderConclusion(results);
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
    if (guess) {
      // the caveat is the report's own sentence, served with the results — one wording for the
      // page, the report and the plot, rather than three that drift apart
      card.appendChild(h("div", { cls: "stat wide" }, [
        h("span", { cls: "pill guess", text: "Best guess · secondary analysis" }),
        h("p", { cls: "caveat", text: results.best_guess_caveat || "" })
      ]));
    }
    card.appendChild(stat(guess ? "best-guess estimate" : "pooled estimate", num(pooled.estimate)));
    var ciLevel = (settings && settings.ci_level !== null && settings.ci_level !== undefined) ? Number(settings.ci_level) : 0.95;
    card.appendChild(stat((ciLevel * 100) + "% CI", "[" + num(pooled.ci_low, 2) + ", " + num(pooled.ci_high, 2) + "]"));
    card.appendChild(stat("k", String(pooled.k),
      "from " + plural(pooled.k_papers || pooled.k, "paper")));
    // I² is a fraction in `pooled.json`, on both lines, and this card is where a reader reads it
    card.appendChild(stat("I²", num(100 * (Number(pooled.I2) || 0), 1) + "%"));
    card.appendChild(stat("τ²", num(pooled.tau2, 3)));
    card.appendChild(stat(guess ? "still held" : "held for review",
      String((guess ? pooled.n_still_held : pooled.n_needs_human) || 0)));
    if (guess) {
      card.appendChild(stat("rows added", String(pooled.n_added || 0),
        pooled.delta_vs_strict === null || pooled.delta_vs_strict === undefined
          ? "" : "moves the estimate by " + num(pooled.delta_vs_strict, 2)));
    }
    card.appendChild(h("div", { cls: "stat wide" }, [
      h("span", { cls: "k", text: "conventions" }),
      h("span", { cls: "v", text: [settings.estimator, settings.variance, settings.tau2_method,
        settings.hakn ? "Hartung-Knapp" : "z intervals",
        "PI " + settings.pi_method].filter(Boolean).join(" · ") })
    ]));
  }

  /* ── the conclusion (DECISION B): the run's own sentences, as text nodes and nothing else ──
     Not written here and not re-derived here: `pooled.json` carries the paragraph the report
     prints, and this card shows those sentences so the page and the file cannot disagree. */
  function renderConclusion(results) {
    var card = $("conclusion-card");
    clear(card);
    var conclusion = (results || {}).conclusion || {};
    var sentences = conclusion.sentences || [];
    if (!sentences.length) { show(card, false); return; }
    // ONE line's headline, the line the reader chose. The paragraph `pooled.json` carries is the
    // whole run's — it names the strict estimate and the best-guess estimate — so showing it
    // whole put both numbers on screen in every state of the toggle, under a comment claiming no
    // state did (review finding 15). The payload names the strict headline sentence and the
    // second line's sentences; whichever line is not on screen has its sentences left out.
    var guess = guessing(results);
    var second = conclusion.best_guess_sentences || [];
    var shown = sentences.filter(function (sentence) {
      return guess ? sentence !== conclusion.headline : second.indexOf(sentence) < 0;
    });
    card.appendChild(h("h2", { text: guess ? "Conclusion · best guess" : "Conclusion" }));
    // …and the caveats ONCE: `conclusion.py` already ends the paragraph with them, and printing
    // `conclusion.caveats` underneath printed every one of them a second time.
    shown.forEach(function (sentence) {
      card.appendChild(h("p", { cls: (conclusion.caveats || []).indexOf(sentence) < 0 ? "" : "hint",
        text: sentence }));
    });
    show(card, true);
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
    var guess = guessing(results);
    var forest = (guess ? results.forest_best_guess : results.forest) || {};
    var renderer = (guess ? results.renderer_best_guess : results.renderer) || {};
    var url = forest.svg;
    if (!url) {
      holder.appendChild(h("p", { cls: "hint",
        text: guess
          ? "No best-guess forest plot: this line has fewer than two rows, or it added none."
          : "No forest plot: fewer than two rows reached the primary analysis." }));
      return Promise.resolve(null);
    }
    // who drew it, and why — the same sentence the report prints under the same figure
    function caption() {
      var parts = [];
      if (guess) { parts.push(results.best_guess_caveat || ""); }
      if (renderer.renderer) {
        parts.push("Drawn by " + renderer.renderer
          + (renderer.reason ? ": " + renderer.reason : "") + ".");
      }
      var text = parts.filter(Boolean).join(" ");
      if (text) { holder.appendChild(h("p", { cls: guess ? "caveat" : "hint", text: text })); }
    }
    return fetch(withToken(url)).then(function (response) { return response.text(); })
      .then(function (text) {
        var svg = sanitizeSvg(text);
        if (!svg) {
          holder.appendChild(h("img", { attrs: { src: withToken(forest.png || url),
            alt: "Forest plot" } }));
          caption();
          return;
        }
        holder.appendChild(svg);
        var hits = makeClickable(svg, results.rows || []);
        if (!hits) {
          holder.appendChild(h("p", { cls: "hint",
            text: "Click a row in the extraction table for its evidence." }));
        }
        caption();
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
    var guess = guessing(results);
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
      // on the best-guess line a held row this line admits is no longer simply "held": it is IN
      // the estimate above, by a named rule, and the reason it is a guess is its title.
      if (!row.in_primary) {
        marks.appendChild(guess && row.in_best_guess
          ? h("span", { cls: "badge guess", text: "best guess",
            attrs: { title: row.best_guess_reason || row.best_guess_rule || "" } })
          : h("span", { cls: "badge", text: "held",
            attrs: { title: guess ? row.best_guess_reason || "" : "" } }));
      }
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
        toast("Recorded as override #" + body.override.seq + ". Re-pool to see it.");
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
      holder.appendChild(h("p", { cls: "hint", text: "Nothing needs your review." }));
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
        "why, e.g. “checked against Table 2, the reading is right”" } });
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
          if (!why) { toast("Give a reason for excluding this dataset."); reason.focus(); return; }
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

    // which line this row is in, and on whose authority (DECISION A): a row can be held out of
    // the primary analysis and still be carrying weight in the guess a reader is looking at.
    var line = record.in_best_guess === true
      ? "in the best guess" + (record.best_guess_rule ? " by rule " + record.best_guess_rule : "")
      : record.in_best_guess === false ? "in neither line" : "";
    body.appendChild(section("What was pooled", [
      kv([
        ["dataset", evidence.dataset_id],
        ["outcome", evidence.outcome_key],
        ["effect", num(record.es) + " [" + num(record.ci_low, 2) + ", " + num(record.ci_high, 2) + "]"],
        ["n (A/B)", (record.n_a || "?") + " / " + (record.n_b || "?")],
        ["route", record.route],
        ["confidence", record.confidence],
        ["best guess", line],
        ["why", record.best_guess_reason],
        ["title", citation.title]
      ])
    ]));

    (evidence.images || []).forEach(function (image) {
      body.appendChild(section("Where it was read · group " + (image.group || "?"), [
        h("img", { cls: "ev-img", attrs: { src: withToken(image.url), loading: "lazy",
          alt: "Page " + (image.page || "?") + " of the paper, with the quoted value highlighted" } }),
        image.quote ? h("blockquote", { cls: "ev-quote", text: image.quote }) : null,
        h("p", { cls: "hint", text: (image.matched ? "" : "the quote could not be located on the page. ")
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
      ["orientation", "Set the direction of this measure"],
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
    var direction = h("select");
    [["false", "a SMALLER raw value is more of what this review scores (error-type measure)"],
     ["true", "a LARGER raw value is more of what this review scores"]].forEach(function (pair) {
      direction.appendChild(h("option", { text: pair[1], attrs: { value: pair[0] } }));
    });
    // H1: a direction belongs to ONE measure. A blank box here used to mean "every measure of
    // this outcome in this paper", which re-signed a different measure's rows and inverted the
    // pooled estimate — so the reviewer picks from the measures the map itself names.
    var measures = (((evidence.dataset || {}).outcomes) || [])
      .filter(function (o) { return o.outcome_key === evidence.outcome_key && o.measure_name; })
      .map(function (o) { return o.measure_name; });
    var measure = h("select");
    if (!measures.length) {
      measure.appendChild(h("option", { text: "the map names no measure for this outcome",
        attrs: { value: "" } }));
    }
    measures.forEach(function (name) {
      measure.appendChild(h("option", { text: name, attrs: { value: name } }));
    });
    var eligible = h("select");
    [["true", "eligible"], ["false", "not eligible"]].forEach(function (pair) {
      eligible.appendChild(h("option", { text: pair[1], attrs: { value: pair[0] } }));
    });
    var justification = h("textarea", { attrs: { rows: 2,
      placeholder: "why: this is the record a reader of your review will check" } });

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
    var directionRow = h("div", { cls: "grid-2" }, [
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "on this measure" }), direction]),
      h("label", { cls: "field" }, [h("span", { cls: "label", text: "measure" }), measure])
    ]);
    hintRow.hidden = true;
    eligibleRow.hidden = true;
    directionRow.hidden = true;

    kind.addEventListener("change", function () {
      valueRow.hidden = kind.value !== "value";
      hintRow.hidden = kind.value !== "re_extract";
      eligibleRow.hidden = kind.value !== "eligibility";
      directionRow.hidden = kind.value !== "orientation";
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
        if (kind.value === "orientation") {
          payload.higher_is_better = direction.value === "true";
          payload.measure_name = (measure.value || "").trim();
          payload.quote = justification.value.trim();
          if (!payload.measure_name) {
            status.textContent = "This outcome's map names no measure, so there is nothing to "
              + "give a direction to.";
            return;
          }
        }
        submit.disabled = true;
        api("/api/runs/" + state.runId + "/overrides", { method: "POST", json: payload })
          .then(function (body) {
            status.textContent = "Recorded as override #" + body.override.seq
              + ". Re-pool to see it in the plot.";
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
            toast(plural(summary.applied, "override") + " applied"
              + (summary.pending.length ? ", " + summary.pending.length + " "
            + (summary.pending.length === 1 ? "needs" : "need") + " a re-run" : ""));
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
      valueRow, hintRow, directionRow, eligibleRow,
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

  /* ───────────────────────────────────────────────────────── paper search
   *
   * The other door into a run: a question instead of a folder. Every shape below is the SERVER's —
   * `canopy/search/models.py` is the contract — because a count this page renamed on its own would
   * render `undefined`, and a phase name it invented would be a rung that never lights.
   *
   * Two rules carried over from the monitor, unchanged: the GET is the truth and the events are an
   * overlay on top of it, and a failure changes what the page SAYS, never what it shows. No list
   * on this screen is ever emptied by an error.
   */

  // COUNT_KEYS, verbatim. The label on screen is the key with its underscores opened out — one
  // word per thing, and a name changed on the server is a name changed here and nowhere else.
  var COUNT_KEYS = ["records", "after_dedupe", "screened", "included", "unsure", "excluded",
                    "excluded_by_user", "not_screened", "fetched", "wanted", "paywalled",
                    "uploaded", "extra", "possible_duplicates"];
  // …and these five are on the strip even at zero: a ladder that grows rungs as it goes hides
  // from the reader what is still to come.
  var COUNTS_ALWAYS = ["records", "after_dedupe", "screened", "included", "fetched"];

  // PHASES, verbatim, with the word each rung is called on screen
  var SEARCH_PHASES = [["queries", "queries"],
                       ["index", "indexes"],
                       ["dedupe", "duplicates"],
                       ["screen", "screening"],
                       ["fetch", "open copies"]];

  // every `CandidateState` lands in exactly one of these. `unscreened` is last on purpose: it is
  // also where a state a newer server invents goes, so a paper can never fall off this page.
  var BUCKETS = ["fetched", "locked", "wanted", "unsure", "excluded", "unscreened", "extra"];

  var STORE_SEARCHES = "canopy.searches";   // { searchId: token }  ← mirrors "canopy.tokens"
  var STORE_SEARCH = "canopy.search";       // the current search id ← mirrors "canopy.current"

  // files this server refused, kept on screen with their reason rather than silently dropped
  var refusedFiles = [];
  /* …and the same for papers `begin` could not read at the last moment. A staged PDF was fetched
     by a machine hours ago; one that no longer probes is left out of the run rather than
     destroying it. That is right, but the page used to report it as a toast carrying only a
     COUNT and then navigate away, so 39 papers could vanish from a review that had already
     started and nothing on any screen would say which. NOTE: the server records the list in
     `job.options` and not in `search.json`, so this survives a navigation and not a reload. */
  var skippedAtBegin = [];
  var beginning = false;        // a `begin` request is in flight: nothing may re-enable the button
  var settledShown = false;     // focus is handed to the title once per search, not on every poll

  function searchOver(search) { return TERMINAL.indexOf((search || {}).status) >= 0; }

  function withSearchToken(url) {
    if (!url) { return ""; }
    return url + (url.indexOf("?") >= 0 ? "&" : "?") + "token="
      + encodeURIComponent(state.searchToken);
  }

  /* api() adds the RUN's bearer only to paths that start "/api/runs" — the path prefix is the
     whole protection — so a search carries its own token and the two can never be swapped. */
  function searchApi(path, opts) {
    opts = opts || {};
    var headers = { Authorization: "Bearer " + state.searchToken };
    Object.keys(opts.headers || {}).forEach(function (k) { headers[k] = opts.headers[k]; });
    return api(path, { method: opts.method, json: opts.json, body: opts.body, headers: headers });
  }

  function searchPath(suffix) {
    return "/api/searches/" + encodeURIComponent(state.searchId) + (suffix || "");
  }

  function paperPath(key, suffix) {
    return searchPath("/papers/" + encodeURIComponent(key) + suffix);
  }

  function rememberSearch(id, token) {
    var tokens = stored(STORE_SEARCHES, {}) || {};
    tokens[id] = token;
    store(STORE_SEARCHES, tokens);
    store(STORE_SEARCH, id);
  }

  function forgetCurrentSearch() { store(STORE_SEARCH, null); }

  function searchTokenFor(id) { return (stored(STORE_SEARCHES, {}) || {})[id] || ""; }

  /* The protocol that is sent is read from the New-run screen at the moment Begin is pressed, so
     an edit made in between is the one that counts. There is no snapshot here to go stale. */
  function protocolText() {
    return state.mode === "yaml" ? $("p-yaml").value : JSON.stringify(guidedProtocol());
  }

  // "is there a protocol at all", asked of whichever editor is in front of the user: #p-title is
  // empty for a pasted YAML that has a perfectly good title of its own.
  function protocolWritten() {
    return (state.mode === "yaml" ? $("p-yaml").value : $("p-title").value).trim() !== "";
  }

  /* ── the mode toggle on the New-run screen ─────────────────────────────── */
  function setPapersFrom(which) {
    state.papersFrom = which === "find" ? "find" : "upload";
    var finding = state.papersFrom === "find";
    Array.prototype.forEach.call(document.querySelectorAll(".src-btn"), function (button) {
      var on = button.getAttribute("data-source") === state.papersFrom;
      button.classList.toggle("is-on", on);
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
    show($("drop"), !finding);
    show($("file-list"), !finding);
    show($("find-panel"), finding);
    show($("dry-run-btn"), !finding);
    show($("run-btn"), !finding);
    show($("find-btn"), finding);
    // #f-max-usd is a number input inside #run-form, so Enter in it fires the form's default
    // button — which is #run-btn, hidden or not. A DISABLED default button is what actually
    // suppresses implicit submission, so mode 2's fields cannot start mode 1 with no files.
    $("run-btn").disabled = finding;
  }

  Array.prototype.forEach.call(document.querySelectorAll(".src-btn"), function (button) {
    button.addEventListener("click", function () {
      setPapersFrom(button.getAttribute("data-source"));
    });
  });

  $("find-btn").addEventListener("click", function () {
    if (state.searchId) { gotoSearch(); return; }   // a second search is started deliberately
    startSearch();
  });

  // exactly the fields `SearchOptions` has: it forbids extras, so a fourth would 422 the search
  function searchOptions() {
    var options = {};
    if ($("f-max-usd").value) { options.max_usd = Number($("f-max-usd").value); }
    if ($("f-max-screened").value) { options.max_screened = Number($("f-max-screened").value); }
    // one entry per line, blanks dropped, spelling untouched: the server reports each line back
    // verbatim, and a line this page tidied would be a line the user never wrote
    var exclude = ($("f-exclude").value || "").split("\n").map(function (line) {
      return line.trim();
    }).filter(function (line) { return line.length > 0; });
    if (exclude.length) { options.exclude = exclude; }
    return options;
  }

  function startSearch() {
    var question = $("draft-sentence").value.trim();
    if (!question) {
      toast("Describe your review in a sentence first.");
      $("draft-sentence").focus();
      return;
    }
    var button = $("find-btn");
    button.disabled = true;
    var body = { question: question, options: searchOptions() };
    // a search needs only the question; beginning the review needs the protocol
    if (protocolWritten()) { body.protocol_text = protocolText(); }
    api("/api/searches", { method: "POST", json: body }).then(function (answer) {
      newSearch(answer.search_id, answer.token);
      rememberSearch(answer.search_id, answer.token);
      gotoSearch();
      listenSearch();
      return refreshSearch();
    }).catch(function (error) { toast(error.message); })
      .then(function () { button.disabled = false; });
  }

  function newSearch(id, token) {
    state.searchId = id;
    state.searchToken = token;
    state.search = null;
    state.searchEvents = [];
    state.pendingDecisions = {};
    refusedFiles = [];
    skippedAtBegin = [];
    beginning = false;
    settledShown = false;
    clear($("search-log"));
    $("find-btn").textContent = "Back to the papers found";
  }

  function gotoSearch() {
    show($("step-search"), true);
    goto("search");
    $("search-title").focus();
  }

  function attachSearch(id, token) {
    newSearch(id, token);
    return refreshSearch().then(function (search) {
      rememberSearch(id, token);
      gotoSearch();
      if (!searchOver(search)) { listenSearch(); }
      return search;
    });
  }

  function refreshSearch() {
    if (!state.searchId) { return Promise.resolve(null); }
    return searchApi(searchPath()).then(function (search) {
      state.search = search;
      renderSearch(search);
      showSearchState(search);
      return search;
    });
  }

  function stopSearchWatchers() {
    if (state.searchSource) { state.searchSource.close(); state.searchSource = null; }
    if (state.searchPoll) { window.clearInterval(state.searchPoll); state.searchPoll = null; }
  }

  function listenSearch() {
    if (state.searchSource) { state.searchSource.close(); }
    var source = new EventSource(withSearchToken(searchPath("/events")));
    state.searchSource = source;
    source.addEventListener("progress", function (message) {
      try { onSearchEvent(JSON.parse(message.data)); } catch (err) { /* a frame we cannot read */ }
    });
    source.addEventListener("end", function (message) {
      var event = {};
      try { event = JSON.parse(message.data); } catch (err) { event = {}; }
      onSearchEvent(event);
      source.close();
      state.searchSource = null;
      refreshSearch().catch(function (error) { toast(error.message); });
    });
    source.onerror = function () { source.close(); state.searchSource = null; };
  }

  function onSearchEvent(event) {
    state.searchEvents.push(event);
    if (event.cost_so_far !== undefined) {
      $("cost-value").textContent = money(event.cost_so_far);
      show($("cost-meter"), true);
    }
    var log = $("search-log");
    log.appendChild(h("li", {}, [
      h("b", { text: phaseLabel(event.stage || "search") }),
      h("span", { cls: "who", text: event.status || "" }),
      h("span", { cls: "msg", text: event.message || "" })
    ]));
    while (log.children.length > 400) { log.removeChild(log.firstChild); }
    log.scrollTop = log.scrollHeight;
    $("search-log-count").textContent = plural(state.searchEvents.length, "event");
    if (event.stage) {
      announce(phaseLabel(event.stage) + ", " + (event.status || "")
               + (event.message ? ". " + event.message : ""));
    }
  }

  /* One sentence per stage at most, and at most one every two seconds: a live region that speaks
     on every event is a screen reader nobody can use. The counts strip is deliberately NOT live —
     it is read on demand — and neither is the log. */
  function announce(text) {
    if (Date.now() - state.announced < 2000) { return; }
    state.announced = Date.now();
    $("search-live").textContent = text;
  }

  function phaseLabel(name) {
    for (var i = 0; i < SEARCH_PHASES.length; i += 1) {
      if (SEARCH_PHASES[i][0] === name) { return SEARCH_PHASES[i][1]; }
    }
    return String(name || "");
  }

  // the pipeline's own words: pending | running | ok | skipped | error — drawn with the monitor's
  // glyphs, so one ladder is one idea across the whole app
  function phaseClass(status) {
    if (status === "ok") { return "dot done"; }
    if (status === "running") { return "dot run"; }
    if (status === "skipped") { return "dot skip"; }
    if (status === "error") { return "dot error"; }
    return "dot";
  }

  /* ── the three progress registers ──────────────────────────────────────── */
  function renderPhases(phases) {
    var list = $("search-phases");
    clear(list);
    (phases || []).forEach(function (phase) {
      var status = phase.status || "pending";
      list.appendChild(h("li", {}, [
        h("span", { cls: phaseClass(status), text: phaseLabel(phase.name) + " — " + status,
                    attrs: { title: phase.message || "" } })
      ]));
    });
  }

  function renderCounts(counts, cost) {
    var strip = $("search-counts");
    clear(strip);
    COUNT_KEYS.forEach(function (key) {
      var value = Number(counts[key] || 0);
      if (!value && COUNTS_ALWAYS.indexOf(key) < 0) { return; }
      strip.appendChild(h("div", { cls: "stat" }, [
        h("span", { cls: "k hint", text: key.replace(/_/g, " ") }),
        h("span", { cls: "v", text: String(value) })
      ]));
    });
    strip.appendChild(h("div", { cls: "stat" }, [
      h("span", { cls: "k hint", text: "spent" }),
      h("span", { cls: "v", text: money(cost) })
    ]));
  }

  function renderFlow(counts, nSources, exclusions) {
    var text = plural(counts.records || 0, "record") + " from " + plural(nSources, "source")
      + " → " + plural(counts.after_dedupe || 0, "unique paper")
      + " → " + plural(counts.screened || 0, "abstract") + " read"
      + " → " + plural(counts.included || 0, "paper") + " wanted"
      + " → " + plural(counts.fetched || 0, "PDF") + " fetched, "
      + plural(counts.paywalled || 0, "paper") + " behind a paywall.";
    if (counts.not_screened) {
      text += " " + plural(counts.not_screened, "record") + " nobody read.";
    }
    if (counts.possible_duplicates) {
      text += " " + plural(counts.possible_duplicates, "pair")
        + " might be the same paper twice; Canopy will not merge those on its own.";
    }
    // the user's own exclusions belong IN the ladder, not beside it: a recall read off this line
    // is not a number until the line says what the search was forbidden to find.
    if (counts.excluded_by_user) {
      text += " " + plural(counts.excluded_by_user, "paper")
        + " you excluded, before anything read or fetched them.";
    }
    var idle = (exclusions || []).filter(function (entry) { return !entry.matched; });
    if (idle.length) {
      // never silent: a mistyped DOI otherwise reads as "this search found none of that paper"
      text += " " + plural(idle.length, "line") + " you excluded matched nothing — "
        + idle.map(function (entry) { return "“" + entry.entry + "”"; }).join(", ")
        + (idle.some(function (entry) { return entry.kind === "refused"; })
           ? " (one of them was too short to use as a title; write the whole DOI)." : ".");
    }
    $("search-flow").textContent = text;
  }

  /* Two cards the markup does not carry, built once on first render and reused. Both exist
     because the server was already sending the data and this page was dropping it on the floor:
     `notes` is where every degradation sentence in the pipeline ends up ("no model was available,
     so nothing was screened", "the screening cap stopped this search", "Unpaywall was not
     consulted"), and the possible-duplicate pairs are the half of the no-auto-merge rule that is
     supposed to cost the reader one click — a click that did not exist. */
  function cardOnce(id, heading, hint) {
    var card = $(id);
    if (card) { return card; }
    card = h("section", { cls: "card", attrs: { id: id, hidden: true,
                                                "aria-labelledby": id + "-h" } }, [
      h("div", { cls: "card-head" }, [
        h("h2", { text: heading, attrs: { id: id + "-h" } }),
        h("span", { cls: "hint", text: hint })
      ]),
      h("ul", { cls: "cand-list", attrs: { id: id + "-list" } })
    ]);
    var before = $("search-log-card");
    before.parentNode.insertBefore(card, before);
    return card;
  }

  function renderNotes(search) {
    var notes = (search.notes || []).filter(function (note) {
      return String(note || "").trim().length > 0;
    });
    var card = cardOnce("search-notes", "What the search noticed",
                        "everything the search wrote down about how it went");
    var list = $("search-notes-list");
    clear(list);
    notes.forEach(function (note) {
      list.appendChild(h("li", { cls: "cand" }, [
        h("span", { cls: "cand-main" }, [h("span", { cls: "cand-why", text: String(note) })])
      ]));
    });
    show(card, notes.length > 0);
  }

  function renderDuplicates(search) {
    var card = cardOnce("search-dupes", "Might be the same paper twice",
                        "Canopy never merges these on its own: a wrong merge deletes a study and "
                        + "nobody ever sees it, while a wrong split costs you one click");
    var list = $("search-dupes-list");
    clear(list);
    var byKey = {};
    (search.candidates || []).forEach(function (candidate) { byKey[candidate.key] = candidate; });
    (search.possible_duplicates || []).forEach(function (pair) {
      var sides = (pair.keys || []).map(function (key) { return byKey[key]; })
        .filter(function (side) { return !!side; });
      if (sides.length !== 2) { return; }          // a pair whose rows are gone is not a question
      var actions = h("span", { cls: "cand-actions" });
      sides.forEach(function (side) {
        if (!pendingKeep(side)) { return; }        // already dropped: the question is answered
        actions.appendChild(h("button", {
          cls: "btn ghost small", text: "Drop " + (side.study_label || side.key),
          attrs: { type: "button" },
          on: { click: function () { dropDuplicate(side); } }
        }));
      });
      list.appendChild(h("li", { cls: "cand" }, [
        h("span", { cls: "cand-main" }, [
          h("strong", { text: sides[0].study_label + " · " + sides[1].study_label }),
          h("span", { cls: "cand-title", text: sides[0].title || "" }),
          h("span", { cls: "cand-title", text: sides[1].title || "" }),
          h("span", { cls: "cand-why", text: pair.why || "" })
        ]),
        actions
      ]));
    });
    show(card, list.children.length > 0);
  }

  /* The one click `models.py` says a false split costs. There is no merge endpoint and there
     should not be one — merging two records is a judgement, and the judgement is the reviewer's;
     dropping the row they decide is the duplicate reaches the same run through the door that
     already exists. */
  function dropDuplicate(candidate) {
    var row = document.querySelector('[data-key="' + candidate.key + '"]');
    return decideCandidate(candidate, false, row || h("li", {}), null)
      .then(function () { redrawCandidates(); });
  }

  function sourceNames(sources) {
    var names = [];
    (sources || []).forEach(function (row) {
      var name = String(row.name || "");
      if (name && names.indexOf(name) < 0) { names.push(name); }
    });
    return names;
  }

  function renderQueries(search) {
    $("query-source-note").textContent = search.query_source === "model"
      ? "written by a model from your question"
      : "built from your own words — no model was used to write them";
    var list = $("search-queries");
    clear(list);
    (search.queries || []).forEach(function (query) {
      list.appendChild(h("li", { cls: "cand" }, [
        h("span", { cls: "cand-main" }, [
          h("span", { cls: "cand-meta", text: query.text || "" }),
          h("span", { cls: "cand-why", text: query.why || "" })
        ])
      ]));
    });
    if (!list.children.length) {
      list.appendChild(h("li", { cls: "cand" }, [h("span", { cls: "hint", text: "No query yet." })]));
    }
    var sources = $("search-sources");
    clear(sources);
    (search.sources || []).forEach(function (row) {
      var badge = row.error
        ? h("span", { cls: "pill stop", text: "no answer" })
        : h("span", { cls: "badge", text: plural(Number(row.n_returned || 0), "record") });
      sources.appendChild(h("li", { cls: "cand" }, [
        h("span", { cls: "cand-main" }, [
          h("strong", { text: row.name || "an index" }),
          h("span", { cls: "cand-meta", text: row.query || row.text || "" }),
          h("span", { cls: "cand-why", text: row.error
            ? String(row.error) : (row.note || "answered") })
        ]),
        h("span", { cls: "cand-actions" }, [badge])
      ]));
    });
  }

  /* ── the lists ─────────────────────────────────────────────────────────── */
  function bucketOf(candidate) {
    var kind = candidate.state;
    if (kind === "fetched" || kind === "uploaded") { return "fetched"; }
    if (kind === "paywalled") { return "locked"; }
    if (kind === "wanted") { return "wanted"; }
    if (kind === "unsure") { return "unsure"; }
    if (kind === "excluded") { return "excluded"; }
    if (kind === "extra") { return "extra"; }
    return "unscreened";
  }

  // a keep/drop whose POST has not answered yet beats the poll, or the tick a person just made
  // flips back under their hand while the server is busy agreeing with it
  function pendingKeep(candidate) {
    var pending = state.pendingDecisions[candidate.key];
    return pending === undefined ? !!candidate.keep : !!pending;
  }

  function metaLine(candidate) {
    var bits = [];
    if (candidate.authors) { bits.push(candidate.authors); }
    if (candidate.venue) { bits.push(candidate.venue); }
    if (candidate.year) { bits.push(String(candidate.year)); }
    if (candidate.doi) { bits.push("doi:" + candidate.doi); }
    return bits.join(" · ");
  }

  /* Canopy's own word for why there is no PDF on a row, in a sentence. `fetch_outcome` is
     recorded for every candidate the fetch stage did not reach, and it was recorded correctly
     and shown nowhere — so a user looking at a paper with no PDF could not tell "no index
     offered a copy" from "the cap stopped us" from "the publisher refused". */
  var FETCH_WORDS = {
    no_oa_location: "No index offered an open-access copy.",
    over_fetch_cap: "The fetch cap stopped this search before this one was tried.",
    cancelled: "You stopped the search before this one was tried.",
    not_wanted: "The screener read it and did not want it, so no copy was fetched.",
    rate_limited: "An index asked us to slow down — that is our request rate, not a paywall.",
    not_a_pdf: "The address we were given did not serve a PDF.",
    http_error: "The publisher refused the download.",
    unreadable: "A file arrived and could not be opened, so it was not kept."
  };

  // every row says why it is in the list it is in — readable, never a tooltip
  function whyLine(candidate) {
    var kind = candidate.state;
    var reason = candidate.reason || "";
    var head = "";
    /* A paper the USER forbade sits in the same list as one the screener threw out, and the two
       must never read alike: the screener's line is a judgement to argue with, this one is an
       instruction that was obeyed. Its own sentence, and no FETCH_WORDS after it — nothing tried
       to fetch this one, so there is no fetch outcome worth explaining. */
    if (candidate.excluded_by_you) {
      return (reason ? reason + ". " : "You excluded this one. ")
        + "Nothing read it, nothing fetched it, and it cost nothing.";
    }
    // deliberately not "Kept:" — `keep` is the reviewer's to change, and a paper nobody screened
    // arrives fetched and unticked, which is the whole point of the keyless path
    if (kind === "fetched") { head = "An open-access copy was downloaded."; }
    else if (kind === "uploaded") { head = "You supplied this PDF."; }
    else if (kind === "paywalled") { head = "No open copy could be fetched, so nothing was."; }
    else if (kind === "wanted") { head = "Wanted. Nothing has tried a publisher yet, so nobody "
      + "may call this one paywalled."; }
    else if (kind === "unsure") { head = "The screener could not tell from the title and abstract."; }
    else if (kind === "excluded") { head = "Ruled out."; }
    else if (kind === "extra") { head = "You added this one; no index proposed it."; }
    else { head = "Nobody read this one: there was no model, or the search stopped first."; }
    if (candidate.title_only) { head += " Only the title was available to read."; }
    if (!candidate.pdf && FETCH_WORDS[candidate.fetch]) {
      head += " " + FETCH_WORDS[candidate.fetch];
    }
    return reason ? head + " " + reason : head;
  }

  function pdfNote(pdf) {
    var bits = [];
    if (pdf.pages) { bits.push(plural(pdf.pages, "page")); }
    if (pdf.bytes) { bits.push((pdf.bytes / 1e6).toFixed(1) + " MB"); }
    return bits.join(" · ");
  }

  /* Every outbound URL is one the SERVER sent. This page cannot build a publisher address or a
     DOI resolver's — its own test forbids the literal — and that is the right rule anyway: a link
     the page invented out of a DOI would be a link nobody audited. */
  function safeLink(link) {
    if (!link || !link.url) { return false; }
    try { return new URL(link.url).protocol === "https:"; } catch (err) { return false; }
  }

  function linkNodes(candidate) {
    return (candidate.links || []).filter(safeLink).slice(0, 2).map(function (link) {
      // deliberately NOT withToken(): this address is a publisher's, not this server's, and a run
      // token on an outbound link would hand a stranger the key to the review. The page's own
      // <meta name="referrer" content="no-referrer"> and rel="noopener noreferrer" do the rest.
      var elsewhere = link.url;
      return h("a", { cls: "btn ghost small", text: link.label || "open",
        attrs: { href: elsewhere, target: "_blank", rel: "noopener noreferrer",
                 "aria-label": "Open " + (link.label || "this paper") + " (new tab)" } });
    });
  }

  function candRow(candidate) {
    var keep = pendingKeep(candidate);
    var name = candidate.study_label || candidate.title || candidate.key;
    var row = h("li", { cls: "cand" + (keep ? "" : " is-dropped"),
                        attrs: { "data-key": candidate.key } });
    var problem = h("span", { cls: "error", attrs: { role: "alert", hidden: true } });
    var main = h("span", { cls: "cand-main" }, [
      h("strong", { text: name }),
      h("span", { cls: "cand-title", text: candidate.title || "" }),
      h("span", { cls: "cand-meta", text: metaLine(candidate) }),
      h("span", { cls: "cand-why", text: whyLine(candidate) }),
      // the abstract excerpt the server sends. Without it a search run with no API key is a list
      // of bare titles: nothing was screened, so no row carries a screener's sentence either, and
      // there is nothing on the page for a person to judge the paper by.
      candidate.abstract_excerpt
        ? h("span", { cls: "cand-meta", text: candidate.abstract_excerpt })
        : null,
      problem
    ]);
    var box = h("input", { attrs: { type: "checkbox", "aria-label": "Keep: " + name } });
    box.checked = keep;
    box.addEventListener("change", function () {
      decideCandidate(candidate, box.checked, row, box);
    });
    row.appendChild(h("label", { cls: "cand-keep" }, [box, main]));

    var actions = h("span", { cls: "cand-actions" });
    if (candidate.source) {
      // provenance gets a word, not a hue
      actions.appendChild(h("span", { cls: "badge", text: candidate.source }));
    }
    if (candidate.state === "fetched" || candidate.state === "uploaded") {
      actions.appendChild(h("span", { cls: "badge ok", text: candidate.state }));
    }
    if (candidate.pdf) { actions.appendChild(h("span", { cls: "hint", text: pdfNote(candidate.pdf) })); }
    // EVERY row with no PDF gets its links and an upload slot, whichever bucket it is in. Offering
    // them only to `locked` and `wanted` is what made a keyless search a dead end: with no model
    // nothing is screened, so every paper lands in `unscreened`, so no row had a link to follow or
    // anywhere to put a PDF — and Begin then refused the search for having nothing readable.
    if (!candidate.pdf) {
      linkNodes(candidate).forEach(function (node) { actions.appendChild(node); });
      actions.appendChild(uploadSlot(candidate, row, problem));
      if (keep) {
        actions.appendChild(h("button", {
          cls: "btn ghost small", text: "Skip", attrs: { type: "button" },
          on: { click: function () { decideCandidate(candidate, false, row, null); } }
        }));
      }
    }
    row.appendChild(actions);
    return row;
  }

  // the input is visually hidden rather than `hidden`, so it keeps its place in the tab order and
  // answers the space bar — a real improvement over the two file inputs on the New-run screen
  function uploadSlot(candidate, row, problem) {
    var input = h("input", { cls: "sr-only", attrs: {
      type: "file", accept: "application/pdf,.pdf",
      "aria-label": "Upload the PDF for " + (candidate.study_label || candidate.key) } });
    input.addEventListener("change", function () {
      if (input.files && input.files.length) {
        uploadCandidate(candidate, input.files[0], row, problem, input);
      }
    });
    return h("label", { cls: "btn small", text: "Upload the PDF" }, [input]);
  }

  function emptyWord(name, search) {
    var counts = search.counts || {};
    if (name === "fetched") {
      return counts.paywalled
        ? "Nothing found was open access. Every paper below is behind a paywall: open each link "
          + "and upload the PDF if you have access."
        : "Nothing has been fetched.";
    }
    if (name === "locked") { return "Nothing was behind a paywall."; }
    if (name === "unsure") { return "The screener was sure about every abstract it read."; }
    if (name === "excluded") { return "Nothing was ruled out."; }
    if (name === "unscreened") {
      // "Every record was read" is only true if a screener read them. With no model this list
      // empties because the open copies were all fetched, and nobody read a word.
      return counts.screened ? "Every record was read." : "No record is waiting here to be read.";
    }
    if (name === "extra") { return "Nothing added by hand."; }
    return "";
  }

  function renderCandidates(search) {
    var groups = {};
    BUCKETS.forEach(function (name) { groups[name] = []; });
    (search.candidates || []).forEach(function (candidate) {
      groups[bucketOf(candidate)].push(candidate);
    });
    BUCKETS.forEach(function (name) {
      var list = $("list-" + name);
      clear(list);
      groups[name].forEach(function (candidate) { list.appendChild(candRow(candidate)); });
      $(name + "-count").textContent = plural(groups[name].length, "paper");
      if (!groups[name].length) {
        list.appendChild(h("li", { cls: "cand" },
                            [h("span", { cls: "hint", text: emptyWord(name, search) })]));
      }
    });
    // a file this server refused is a fact on the page, not a message that scrolled past
    refusedFiles.forEach(function (bad) {
      $("list-extra").appendChild(h("li", { cls: "cand" }, [
        h("span", { cls: "cand-main" }, [
          h("strong", { text: bad.filename || "a file" }),
          h("span", { cls: "error", text: bad.reason || "this server refused it" })
        ])
      ]));
    });
    show($("card-wanted"), groups.wanted.length > 0);
    show($("card-unscreened"), groups.unscreened.length > 0);
    // …and it ships collapsed, which is right when a screener read everything and three records
    // slipped through, and wrong when nothing was screened at all: that is the whole result set,
    // and a user who has to find a disclosure triangle to see their search has not been shown it.
    if (!(search.counts || {}).screened && groups.unscreened.length) {
      $("card-unscreened").open = true;
    }
  }

  /* ── keep, drop, upload ────────────────────────────────────────────────── */
  function decideCandidate(candidate, keep, row, box) {
    var key = candidate.key;
    state.pendingDecisions[key] = keep;
    row.classList.toggle("is-dropped", !keep);
    // the promise is returned so a caller that is not the checkbox — the duplicate card's one
    // click — can redraw once the server has agreed, instead of guessing when to
    return searchApi(paperPath(key, "/decide"), { method: "POST", json: { keep: keep } })
      .then(function (answer) {
        candidate.keep = !!answer.keep;
        delete state.pendingDecisions[key];
        if (state.search) { renderBeginBar(state.search); }
      })
      .catch(function (error) {
        delete state.pendingDecisions[key];
        row.classList.toggle("is-dropped", !candidate.keep);
        if (box) { box.checked = !!candidate.keep; }
        toast(error.message);
      });
  }

  // what the browser can know before a round trip; everything else is the server's to refuse, and
  // its own words are shown verbatim rather than paraphrased
  function badPdf(file) {
    if (!/\.pdf$/i.test(file.name)) { return "That file is not a PDF. Canopy only reads PDFs."; }
    var cap = (state.settings || {}).max_upload_mb;
    if (cap && file.size > cap * 1e6) {
      return "That PDF is " + (file.size / 1e6).toFixed(1) + " MB; this server accepts up to "
        + cap + " MB.";
    }
    return "";
  }

  function showRowError(problem, message, input) {
    problem.textContent = message;
    show(problem, true);
    if (input) { input.focus(); }
  }

  function uploadCandidate(candidate, file, row, problem, input) {
    var refuse = badPdf(file);
    if (refuse) { showRowError(problem, refuse, input); return; }
    show(problem, false);
    input.disabled = true;
    var was = row.querySelector(".cand-actions");
    if (was) { was.appendChild(h("span", { cls: "dot run", text: "uploading…" })); }
    var form = new FormData();
    form.append("file", file, file.name);
    searchApi(paperPath(candidate.key, "/upload"), { method: "POST", body: form })
      .then(function (updated) {
        mergeCandidate(updated);
        redrawCandidates();
        toast("Matched to " + (updated.study_label || file.name)
              + ". It joins the review as an uploaded PDF.");
      })
      .catch(function (error) {
        input.disabled = false;
        showRowError(problem, error.message, input);
      });
  }

  function mergeCandidate(updated) {
    if (!state.search) { return; }
    var rows = state.search.candidates || [];
    var found = false;
    for (var i = 0; i < rows.length; i += 1) {
      if (rows[i].key === updated.key) { rows[i] = updated; found = true; }
    }
    if (!found) { rows.push(updated); }
    state.search.candidates = rows;
  }

  function redrawCandidates() {
    if (!state.search) { return; }
    renderCandidates(state.search);
    renderBeginBar(state.search);
    // the counts are the server's to state, so they are re-read rather than incremented here
    refreshSearch().catch(function () { /* the rows are already right; the counts follow */ });
  }

  function uploadExtras(fileList) {
    var files = Array.prototype.filter.call(fileList || [], function (file) {
      return /\.pdf$/i.test(file.name);
    });
    var note = $("search-extra-note");
    if (!files.length) {
      note.textContent = "That file is not a PDF. Canopy only reads PDFs.";
      return;
    }
    var form = new FormData();
    files.forEach(function (file) { form.append("files", file, file.name); });
    note.textContent = "Uploading " + plural(files.length, "PDF") + "…";
    searchApi(searchPath("/papers"), { method: "POST", body: form })
      .then(function (answer) {
        var added = answer.added || [];
        var refused = answer.rejected || [];
        added.forEach(function (candidate) { mergeCandidate(candidate); });
        refused.forEach(function (bad) { refusedFiles.push(bad); });
        note.textContent = plural(added.length, "paper") + " added"
          + (refused.length ? " · " + plural(refused.length, "file") + " refused, each with its "
             + "reason below" : "") + ".";
        redrawCandidates();
      })
      .catch(function (error) { note.textContent = error.message; });
  }

  $("search-extra-input").addEventListener("change", function (event) {
    uploadExtras(event.target.files);
  });

  var searchDrop = $("search-drop");
  ["dragenter", "dragover"].forEach(function (name) {
    searchDrop.addEventListener(name, function (event) {
      event.preventDefault(); searchDrop.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    searchDrop.addEventListener(name, function (event) {
      event.preventDefault(); searchDrop.classList.remove("is-over");
    });
  });
  searchDrop.addEventListener("drop", function (event) {
    if (event.dataTransfer && event.dataTransfer.files) { uploadExtras(event.dataTransfer.files); }
  });

  /* ── what the page says when something went wrong ──────────────────────── */
  function searchBanner(search, over) {
    var counts = search.counts || {};
    var out = [];
    // the cap is keyed on `stopped_because`, NOT on the status: a capped search still finishes
    // `done`, and a user told only "done" would never learn their search was cut short
    if (search.stopped_because === "budget") {
      out.push("The search reached its spending cap after reading "
        + plural(counts.screened || 0, "abstract") + " of " + (counts.after_dedupe || 0)
        + ". The rest are under Not screened — nothing was thrown away.");
    } else if (search.stopped_because === "deadline") {
      out.push("The search ran out of time. What it had already found is here, and the records it "
        + "never read are under Not screened — nothing was thrown away.");
    } else if (search.stopped_because === "cancelled" || search.status === "cancelled") {
      out.push("Search stopped. " + plural(counts.fetched || 0, "PDF")
        + " had already been fetched and are still here.");
    }
    if (search.status === "interrupted") {
      // NOT "nothing was lost": a search that dies with the server keeps the PDFs already on
      // disk and loses the screening decisions and the candidate list that had not been
      // written yet. Saying otherwise was the page telling the user something the server
      // could not support — the recovery pass explains what survived, and `notes` (below)
      // carries the server's own account of what did not.
      out.push("This search stopped when the server did. "
        + plural(counts.fetched || 0, "paper") + " already fetched are still here; anything the "
        + "search had not written down when it stopped is gone. Search again to look for the rest.");
    }
    if (search.status === "error" && search.error) { out.push(String(search.error)); }
    var dead = (search.sources || []).filter(function (row) { return row.error; });
    if (dead.length) {
      var live = sourceNames((search.sources || []).filter(function (row) { return !row.error; }));
      out.push((live.length ? live.join(", ") + " answered. " : "")
        + sourceNames(dead).join(", ") + " did not. The papers below are from the indexes that "
        + "answered, so this search is narrower than it looks.");
    }
    if (over && !(search.candidates || []).length) {
      out.push("No records matched. The queries and what each index answered are under “What the "
        + "search did” — try naming the population or the outcome you want.");
    } else if (over && !counts.fetched && !counts.uploaded && counts.paywalled) {
      out.push("Nothing found was open access. Every paper below is behind a paywall: open each "
        + "link and upload the PDF if you have access.");
    }
    if (search.run_id) {
      out.push("This search has already become a review, so its papers are fixed.");
    }
    skippedAtBegin.forEach(function (bad) {
      out.push("Left out of the review: " + (bad.filename || "one paper") + " — "
        + (bad.reason || "it could not be read when the run was built") + ".");
    });
    return out;
  }

  function bannerActions(search, over) {
    var buttons = [];
    if (!over) { return buttons; }
    if (search.stopped_because === "budget") {
      buttons.push(["Search again with a higher cap", function () { searchAgain(true); }]);
    }
    if (!(search.candidates || []).length) {
      buttons.push(["Edit the question", function () {
        goto("new"); setPapersFrom("find"); $("draft-sentence").focus();
      }]);
    }
    if (!buttons.length && (search.status === "interrupted" || search.stopped_because)) {
      buttons.push(["Search again", function () { searchAgain(false); }]);
    }
    return buttons;
  }

  function showSearchState(search) {
    var over = searchOver(search);
    var stop = $("search-cancel-btn");
    stop.disabled = over;
    stop.textContent = over ? "Search " + search.status : "Stop the search";
    if (over) { stop.setAttribute("title", "this search has already finished"); }
    else { stop.removeAttribute("title"); }

    show($("search-results"), over);
    $("search-log-card").open = !over;

    var banner = $("search-banner");
    clear(banner);
    var words = searchBanner(search, over);
    show(banner, words.length > 0);
    words.forEach(function (word) { banner.appendChild(h("span", { text: word + " " })); });
    bannerActions(search, over).forEach(function (pair) {
      banner.appendChild(h("button", { cls: "btn ghost small", text: pair[0],
        attrs: { type: "button" }, on: { click: pair[1] } }));
    });

    // focus moves to the title once, and only if the reviewer is still on this screen: a person
    // typing somewhere else must never have the caret taken off them
    if (over && !settledShown) {
      settledShown = true;
      if ($("screen-search").contains(document.activeElement)) { $("search-title").focus(); }
    }
    pollSearchWhileRunning(search);
  }

  /* A search outlives the page that started it: while one is going the screen asks the server for
     the whole record every few seconds, so a dropped stream or a sleeping tab costs nothing. */
  function pollSearchWhileRunning(search) {
    var going = !searchOver(search);
    if (going && !state.searchPoll) {
      state.searchPoll = window.setInterval(function () {
        refreshSearch().catch(function () { /* a transient failure is not worth a toast */ });
      }, 5000);
    }
    if (!going && state.searchPoll) {
      window.clearInterval(state.searchPoll);
      state.searchPoll = null;
    }
  }

  function renderSearch(search) {
    var counts = search.counts || {};
    renderPhases(search.phases || []);
    renderCounts(counts, search.cost_usd);
    renderFlow(counts, sourceNames(search.sources).length, search.exclusions);
    renderQueries(search);
    renderNotes(search);
    renderDuplicates(search);
    renderCandidates(search);
    renderBeginBar(search);
    $("run-title").textContent = "Paper search · " + (search.search_id || "");
    if (search.cost_usd) {
      $("cost-value").textContent = money(search.cost_usd);
      show($("cost-meter"), true);
    }
  }

  /* ── the begin bar ─────────────────────────────────────────────────────── */
  function keptPapers(search) {
    return (search.candidates || []).filter(function (candidate) {
      return pendingKeep(candidate) && candidate.pdf;
    });
  }

  // papers that have a PDF, ticked or not. The difference between "there is nothing to run on"
  // and "there is, and you have not chosen it yet" is two different instructions, and a keyless
  // search reaches the second one with every row unticked.
  function readablePapers(search) {
    return (search.candidates || []).filter(function (candidate) { return !!candidate.pdf; });
  }

  function renderBeginBar(search) {
    var over = searchOver(search);
    var kept = keptPapers(search);
    var byState = function (name) {
      return kept.filter(function (c) { return c.state === name; }).length;
    };
    show($("begin-bar"), over);

    var readable = readablePapers(search);
    $("begin-papers").textContent = kept.length
      ? plural(kept.length, "paper") + " ticked and readable · " + byState("fetched")
        + " fetched, " + byState("uploaded") + " you uploaded, " + byState("extra")
        + " you added · this creates the run and starts it."
      : (readable.length
         ? "Nothing is ticked. " + plural(readable.length, "paper") + " here can be read — tick "
           + "the ones you want in the review."
         : "Upload at least one PDF to begin. A run needs a paper it can actually read.");
    $("begin-protocol").textContent = protocolWritten()
      ? (state.mode === "yaml"
         ? "Protocol: the YAML on the New run screen."
         : "Protocol: " + $("p-title").value.trim() + " · "
           + plural((guidedProtocol().outcomes || []).length, "outcome")
           + " · profile " + $("p-profile").value)
      : "No protocol yet. A review needs one — write it on the New run screen.";
    // MIRRORED from the one Advanced block, never copied: two sets of inputs is two truths
    var options = runOptions();
    $("begin-options").textContent =
      (options.budget_usd ? money(options.budget_usd) + " budget" : "No budget cap")
      + " · " + (options.max_usd_per_paper ? money(options.max_usd_per_paper) + " per paper"
                 : "no per-paper cap")
      + " · " + plural(options.concurrency, "paper") + " at a time"
      + (options.max_papers ? " · the first " + plural(options.max_papers, "paper") + " only" : "");

    var button = $("begin-btn");
    if (beginning) { button.disabled = true; return; }
    if (search.run_id) {
      button.disabled = true;
      button.textContent = "Already a review";
      button.setAttribute("title", "this search became a review; start another to look again");
      return;
    }
    button.removeAttribute("title");
    if (!over) {
      button.disabled = true;
      button.setAttribute("title",
        "Still searching. The button opens when the search is done or you stop it.");
    } else if (!kept.length) {
      button.disabled = true;
      button.setAttribute("title", readable.length
        ? "Tick the papers you want. A run needs at least one it can read."
        : "Upload at least one PDF to begin.");
    } else if (!protocolWritten()) {
      button.disabled = false;
      button.textContent = "Write the protocol";
      button.classList.add("pending");
    } else {
      button.disabled = false;
      button.textContent = "Run the review";
      button.classList.remove("pending");
    }
  }

  /* #begin-btn is not a submit, so browser validation never fires here — and asking #run-form to
     validate itself would try to focus a control on a screen nobody can see, which is the exact
     failure `demandFields` exists for. The protocol is checked in JS instead. */
  function beginRun() {
    var button = $("begin-btn");
    if (!protocolWritten()) {
      toast("A protocol needs a title. Write the protocol, then begin.");
      goto("new");
      $("p-title").focus();
      button.classList.add("pending");
      return;
    }
    button.classList.remove("pending");
    beginning = true;
    button.disabled = true;      // begin re-reads every staged PDF; two clicks would be two runs
    var options = runOptions();
    options.start = true;
    searchApi(searchPath("/begin"),
              { method: "POST", json: { protocol_text: protocolText(), options: options } })
      .then(function (body) {
        skippedAtBegin = (body.skipped || []).slice();
        if (skippedAtBegin.length) {
          // named, not counted, and written onto the search screen as well as spoken once: a
          // review that quietly started without 39 of its 40 papers is not a toast
          toast(plural(skippedAtBegin.length, "paper")
                + " could not be read at the last moment and was left out of the run: "
                + skippedAtBegin.map(function (bad) { return bad.filename || "one paper"; })
                  .join(", ") + ".");
          if (state.search) { showSearchState(state.search); }
        }
        stopSearchWatchers();
        forgetCurrentSearch();      // the run claims the page now; the search stays openable
        // …and from here it is an ordinary run: attach() remembers the token, paints the
        // manifest, opens the stream and navigates. There is no second monitor.
        return attach(body.run_id, body.token);
      })
      .catch(function (error) {
        beginning = false;
        button.disabled = false;
        toast(error.message);
      });
  }

  function searchAgain(higher) {
    stopSearchWatchers();
    if (higher) {
      var was = Number($("f-max-usd").value) || 0;
      $("f-max-usd").value = (was ? was * 2 : 2).toFixed(2);
    }
    state.searchId = "";
    state.searchToken = "";
    state.search = null;
    forgetCurrentSearch();
    $("find-btn").textContent = "Find papers";
    goto("new");
    setPapersFrom("find");
    $("f-max-usd").focus();
  }

  $("begin-btn").addEventListener("click", beginRun);
  $("search-again-btn").addEventListener("click", function () { searchAgain(false); });
  $("begin-edit-btn").addEventListener("click", function () {
    goto("new");
    $("p-title").focus();
  });

  $("search-cancel-btn").addEventListener("click", function () {
    if (!state.searchId) {
      toast("There is no search to stop yet.");
      return;
    }
    searchApi(searchPath("/cancel"), { method: "POST" })
      .then(function (body) {
        toast("Search " + body.status + ". Finished work is saved.");
        return refreshSearch();
      })
      .catch(function (error) { toast(error.message); });
  });

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

  // …and the same for a search, but only when no run claims the page: a run is further along than
  // the search that made it. A search id the server no longer has is forgotten in silence — a
  // dead id is not news, and it is exactly what a run does today.
  if (!saved || !tokenFor(saved)) {
    var savedSearch = stored(STORE_SEARCH, "");
    var savedSearchToken = savedSearch ? searchTokenFor(savedSearch) : "";
    if (savedSearchToken) {
      attachSearch(savedSearch, savedSearchToken).catch(function () { forgetCurrentSearch(); });
    } else {
      // Nothing remembered here — but a search the user PAID for may still be on the server,
      // and a search this browser cannot name is a search they cannot get back to. On a
      // loopback server the listing carries its own tokens (the runs list does the same), so
      // the most recent unfinished search is re-attachable after a reload, a new tab, or a
      // browser that forgot its storage. On a remote server no token comes back and nothing
      // is claimed: silence is correct there, not an error worth showing.
      api("/api/searches").then(function (body) {
        var rows = (body && body.searches) || [];
        var mine = rows.filter(function (row) { return row.token && !row.run_id; });
        if (mine.length) {
          attachSearch(mine[0].search_id, mine[0].token).catch(function () {});
        }
      }).catch(function () { /* no listing, no news */ });
    }
  }
})();
