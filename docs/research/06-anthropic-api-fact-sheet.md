# Anthropic (Claude) API fact sheet for Canopy — verified 2026-08-14

Scope: every number, parameter name and limit below was read from the **live** docs at
`platform.claude.com/docs/en/...` (fetched as raw `.md`) or the `anthropics/anthropic-sdk-python`
`main` branch (SDK 0.122.0 on PyPI, `requires-python >= 3.9`) on 2026-08-14. Anything I could not
confirm on a primary page is tagged **[UNVERIFIED]**. Section-level source URLs are at the end.

Canopy-relevant framing: our workload is (a) many research-paper PDFs (10–40 pages each), (b) figure
crops that must be read as images, (c) strict JSON extraction with page-level provenance, (d) many
independent verify/vote calls (batch-friendly), and (e) a fixed protocol prompt reused across every
paper (cache-friendly).

---

## 1. Models and pricing (Messages API, first-party)

Source: models overview + pricing pages.

| Model | ID / alias | Context | Max output | $/MTok in / out | Thinking default | Effort levels | Notes |
|---|---|---|---|---|---|---|---|
| Claude Fable 5 | `claude-fable-5` | 1M | 128k | $10 / $50 | **always on** (adaptive); `{type:"disabled"}` -> 400 | low, medium, high, xhigh, max | Same tokenizer as Opus 4.7/4.8 (~30% more tokens than pre-4.7 models). Safety classifiers can return `stop_reason:"refusal"`. Requires 30-day data retention (not ZDR-eligible; per skill/migration guide — [UNVERIFIED on live page]). |
| Claude Opus 5 | `claude-opus-5` | 1M | 128k | $5 / $25 | on (adaptive) when `thinking` omitted; `disabled` allowed only at effort `high` or lower (400 at `xhigh`/`max`) | low, medium, high, xhigh, max | Recommended default. Refusal classifiers apply. Separate rate-limit bucket from Opus 4.x. |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | 128k | **$2 / $10** (the "introductory" price is now permanent per the pricing page note) | on (adaptive) when omitted; `disabled` allowed | low, medium, high, xhigh, max | New tokenizer (~30% more tokens vs Sonnet 4.6). High-res vision tier. |
| Claude Haiku 4.5 | `claude-haiku-4-5` (= `claude-haiku-4-5-20251001`) | 200k | 64k | $1 / $5 | off; extended thinking via `{type:"enabled", budget_tokens:N}` only (no adaptive) | **not supported** (effort errors on Haiku 4.5) | Standard vision tier (1568px). |

* `effort` lives in `output_config.effort`; the API default is `high` for Opus 5 / Sonnet 5 / Opus 4.8; setting `"high"` explicitly is identical to omitting it. `xhigh` is documented on Fable 5, Mythos 5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5. No beta header.
* Fable 5 / Opus 5 / Sonnet 5 / Opus 4.8 / 4.7 reject `thinking:{type:"enabled",budget_tokens:N}` and (Opus 5/4.8/4.7/Fable/Sonnet 5) reject `temperature`/`top_p`/`top_k` (400). Assistant-turn prefill returns 400 on all 4.6+ models.
* Thinking `display` defaults to `"omitted"` on Fable 5, Opus 5, Sonnet 5, Opus 4.8/4.7 (empty `thinking` text, signature still present); pass `thinking={"type":"adaptive","display":"summarized"}` to get readable summaries. Raw chain-of-thought is never returned.
* Prompt-cache multipliers (all models): 5-min write 1.25x, 1-h write 2x, read 0.1x. E.g. Opus 5: $6.25 / $10 / $0.50 per MTok; Sonnet 5: $2.50 / $4 / $0.20; Haiku 4.5: $1.25 / $2 / $0.10; Fable 5: $12.50 / $20 / $1.
* Batch = 50% of everything (input, output, and it stacks with cache multipliers): Opus 5 $2.50/$12.50; Sonnet 5 $1/$5; Haiku 4.5 $0.50/$2.50; Fable 5 $5/$25.
* Long context: 1M window at standard price, no long-context premium (4.6+ models).
* Server tools: web search $10 per 1,000 searches; code execution 1,550 free container-hours/month/org then $0.05/container-hour (free when `web_search_20260209`/`web_fetch_20260209` is in the request).
* Fast mode (`speed:"fast"`, beta `fast-mode-2026-02-01`): Opus 5 / Opus 4.8 only, $10/$50, not in Batches. Not relevant for Canopy.
* Models API: `client.models.retrieve("claude-opus-5")` returns `max_input_tokens`, `max_tokens`, `capabilities` (query at runtime instead of hard-coding).

Knowledge cutoffs (reliable): Fable 5 Jan 2026, Opus 5 May 2026, Sonnet 5 Jan 2026, Haiku 4.5 Feb 2025.

---

## 2. Rate limits by tier (Messages API, first-party)

Source: `api/rate-limits`. Limits are per model class; **only uncached input tokens count toward ITPM** (`input_tokens` + `cache_creation_input_tokens`; `cache_read_input_tokens` do NOT count). OTPM counts actual output tokens; `max_tokens` does not affect OTPM. New orgs may start in an "Evaluation tier" below these numbers.

| Tier (monthly spend cap) | Model | RPM | ITPM | OTPM |
|---|---|---|---|---|
| Start ($500) | Fable 5 | 1,000 | 500,000 | 100,000 |
| Start | Opus 5 | 1,000 | 2,000,000 | 400,000 |
| Start | Sonnet 5 | 1,000 | 2,000,000 | 400,000 |
| Start | Haiku 4.5 | 1,000 | 2,000,000 | 400,000 |
| Build ($1,000) | Fable 5 | 2,000 | 1,500,000 | 300,000 |
| Build | Opus 5 / Sonnet 5 / Haiku 4.5 | 5,000 | 5,000,000 | 1,000,000 |
| Scale ($200,000) | Fable 5 | 4,000 | 4,000,000 | 800,000 |
| Scale | Opus 5 / Sonnet 5 / Haiku 4.5 | 10,000 | 10,000,000 | 2,000,000 |

* Opus 5 and Sonnet 5 each have their **own** bucket (not shared with the combined Opus 4.x / Sonnet 4.x pools).
* Message Batches API (shared across models): Start 1,000 RPM / 200,000 batch requests in queue / 100,000 per batch; Build 2,000 / 300,000 / 100,000; Scale 4,000 / 500,000 / 100,000.
* Token counting (`POST /v1/messages/count_tokens`): free; separate RPM pool: Start 2,000, Build 4,000, Scale 8,000.
* Files API during beta: ~100 requests/minute.
* 429s carry `retry-after`; headers `anthropic-ratelimit-{requests,input-tokens,output-tokens}-{limit,remaining,reset}`. SDK retries 408/409/429/5xx twice by default (`max_retries=2`).

Practical: at Start tier, 2M ITPM on Opus 5 = ~40 full 40-page papers (~50k tokens each incl. page images) per minute if uncached. Caching the protocol/system prompt keeps those tokens off ITPM.

---

## 3. PDF input

Source: `build-with-claude/pdf-support`.

| Fact | Value |
|---|---|
| Max request size | **32 MB** (entire JSON payload incl. base64) — Messages & count_tokens; Batches 256 MB; Files upload 500 MB. 413 `request_too_large` if exceeded. |
| Max pages per request | **600** for 1M-context models; **100** when the request's context window is under 1M (i.e. Haiku 4.5). Limit is across all PDFs in the request. |
| Format | Standard PDF, no password/encryption |
| Models | All active models |
| Beta header | none for base64/url; **`files-api-2025-04-14`** when using `file_id` |
| Source types | `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":...}}` \| `{"type":"url","url":...}` \| `{"type":"file","file_id":...}` |
| Processing | "The system converts each page of the document into an image. The text from each page is extracted and provided alongside each page's image." Claude sees text + image per page. |
| Token cost | Text: "Each page typically uses 1,500–3,000 tokens per page depending on content density" **plus** image tokens per page computed with the vision formula (see §4). No extra PDF fee. Use `count_tokens` for exact numbers. |
| Placement | "Place PDFs before text in your requests"; use logical (viewer) page numbers in prompts; split large PDFs. |
| Dense PDFs | Can exhaust context or fail before the page limit even via Files API; split into sections / downsample embedded images. |
| Citations | Supported: `"citations":{"enabled":true}` on the document block -> `page_location` citations with `start_page_number` (1-indexed) and `end_page_number` (exclusive). Citing images inside PDFs is **not** supported (text only). |
| Coordinates | Pages are rasterized server-side at dimensions you do not control, so bbox coordinates on PDF pages **cannot be mapped back** — rasterize pages yourself (PyMuPDF) and send images if you need pixel bboxes (vision-coordinates page). |
| Caching / Batches | `cache_control` may be placed on the document block; PDFs work inside Batches. |
| Other formats | `.txt/.csv/.md` -> `document` with `text/plain`; `.docx/.xlsx` are not accepted in `document` blocks (convert to PDF/text). |

Canopy implication: sending a whole 30-page paper as one `document` costs roughly 30 × (1.5–3k text + ~1.5–2.7k image) ≈ 90–170k tokens. Cheaper and more controllable: extract text ourselves (PyMuPDF), send only the pages/figures that matter as images (§4), and keep our own page indices for provenance.

---

## 4. Vision (images)

Source: `build-with-claude/vision` + `vision-coordinates`.

| Fact | Value |
|---|---|
| Source types | `base64`, `url`, `file` (`file_id`, needs `files-api-2025-04-14`) |
| Formats | JPEG, PNG, GIF, WebP (`image/jpeg`, `image/png`, `image/gif`, `image/webp`); animations: first frame only |
| Max images per request | **600** (1M-context models); **100** for 200k-context models (Haiku 4.5); 20 per message on claude.ai |
| Max per-image size | 10 MB base64 (Claude API); max dimensions 8000×8000 px |
| >20 images in one request | stricter per-image dimension limit; keep every dimension ≤ 2000 px or ≤ 20 image+document blocks |
| Token formula | **`⌈width/28⌉ × ⌈height/28⌉` visual tokens** (28×28-pixel patches). (The old `w×h/750` rule is gone.) |
| Resolution tiers | **High-res (Claude 4.7 and later — Opus 5, Sonnet 5, Fable 5, Opus 4.8/4.7): max long edge 2576 px, max 4784 visual tokens.** Standard (all others incl. Haiku 4.5): 1568 px / 1568 tokens. Automatic, no header. |
| Downscale rule | Image resized (aspect preserved) until both (a) neither edge > tier edge limit and (b) `⌈w/28⌉×⌈h/28⌉` ≤ tier token limit; then padded to a multiple of 28 on right/bottom. Reference implementation on the coordinates page (uses banker's rounding). Examples: 1920×1080 -> 1456×819 (1560 tok) standard / unchanged (2691 tok) high-res; 3840×2160 -> 2576×1449 (4784 tok) high-res. An A4 scan at 130 DPI (1075×1520 = 2145 tok) is resized on standard tier but not on high-res. |
| Cost examples (docs) | 1000×1000 image = 1296 tokens ≈ $6.48 per 1,000 images on Opus 5; a 4K image ≈ $23.92 per 1,000 on Opus 5. |
| Coordinates | Ask for **absolute pixel coordinates** `[x1,y1,x2,y2]` (not 0–1000 normalized); origin top-left; coords refer to the resized (un-padded) image; pre-resize yourself so they map 1:1. Crop regions of interest for small targets. Structured outputs recommended for bbox JSON. |
| Ordering | Images before text; label multiple images ("Image 1:", "Image 2:"). |
| Limitations | approximate counting/localization; low-quality or <200 px images error-prone. |
| Metadata | Not read. Images ephemeral (deleted after processing) unless uploaded via Files API. |

Canopy implication: render figure crops at ≤ 2576 px long edge and ≤ 4784 tokens on Opus 5/Sonnet 5 (e.g. a 1600×1200 crop = 58×43 = 2494 tokens ≈ $0.012 on Opus 5, $0.005 on Sonnet 5). Zoom = crop, not upscale.

---

## 5. Citations

Source: `build-with-claude/citations`. All active models. No beta header.

* Enable per document: `"citations": {"enabled": true}` on each `document` block — **all or none** of the documents in a request. Optional `title` (short) and `context` (free text/JSON metadata, not citable).
* Chunking: PDF and plain text -> sentence chunks (Claude can chain consecutive sentences); custom content (`source.type:"content"` with a list of blocks) -> your blocks are the citation units.
* Citation objects (inside `text` blocks' `citations` array):
  * PDF: `{"type":"page_location","cited_text":..., "document_index":0, "document_title":..., "start_page_number":1 (1-indexed), "end_page_number":2 (exclusive)}`
  * plain text: `char_location` with `start_char_index` (0-indexed) / `end_char_index` (exclusive)
  * custom content: `content_block_location` with `start_block_index` / `end_block_index` (exclusive)
  * `document_index` is 0-indexed across all document blocks in the request (all messages).
* Cost: `cited_text` does **not** count as output tokens (nor input when passed back); enabling citations slightly increases input tokens.
* Streaming: citations arrive as `citations_delta` inside `content_block_delta`.
* Works with prompt caching (cache the document block, not the citation), token counting, batches, Files API `file_id` documents.
* **Incompatible with structured outputs**: citations enabled + `output_config.format` -> **400**. Also cannot cite images inside PDFs.

Canopy implication: two-pass design — pass A (citations on, free-text answer with page-anchored quotes) or pass B (structured JSON, no citations, but ask the model to echo `page` + verbatim `quote` fields that we validate against our own PyMuPDF text). We cannot have both in one call.

---

## 6. Files API (beta)

Source: `build-with-claude/files`. Beta header **`files-api-2025-04-14`** on upload AND on every Messages request that references a `file_id` (SDK adds it automatically for `client.beta.files.*`; pass `betas=["files-api-2025-04-14"]` to `client.beta.messages.create`).

| Fact | Value |
|---|---|
| Endpoints | `POST /v1/files` (multipart `file=@...`), `GET /v1/files` (paginated, `limit` default 20, `before_id`/`after_id`), `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, `DELETE /v1/files/{id}` |
| Python SDK | `client.beta.files.upload(file=("name.pdf", open(...,"rb"), "application/pdf"))` or `file=Path(...)`; `.list()`, `.retrieve_metadata(id)`, `.delete(id)`, `.download(id)` -> `BinaryAPIResponse` (`.write_to_file(path)`) |
| Response | `{"id":"file_...","type":"file","filename","mime_type","size_bytes","created_at","downloadable":false}` |
| Limits | 500 MB per file; **500 GB** per organization; filenames 1–255 chars, no `< > : " \| ? * \ /` |
| Lifecycle | persist until deleted; immutable; workspace-scoped (any key in the workspace can read any file — never accept `file_id` from users) |
| Content blocks | PDF (`application/pdf`) & `text/plain` -> `{"type":"document","source":{"type":"file","file_id":...}, "title","context","citations"}`; images -> `{"type":"image","source":{"type":"file","file_id":...}}`; datasets/other -> `{"type":"container_upload","file_id":...}` for the code-execution container |
| Download | only files created by code execution/skills (`downloadable:true`); downloading an uploaded file -> 400 |
| Billing | all file operations free; content is billed as input tokens when used |
| Availability | Claude API (beta), Claude Platform on AWS, Foundry; **not** Bedrock/Vertex; not ZDR-eligible |
| Batches | file_id documents work in batch requests (the batch doc lists "most beta features"; PDF page shows a batch example) — **[UNVERIFIED that Batches API accepts the files beta header on `client.messages.batches`; use `client.beta.messages.batches.create(betas=[...])`]** |

Canopy implication: upload each PDF once (`file_id`), then reference it from extraction, verification and adjudication calls; keeps each request far below 32 MB and avoids re-encoding base64.

---

## 7. Message Batches API

Source: `build-with-claude/batch-processing`. No beta header. Python: `client.messages.batches.create(requests=[Request(custom_id=..., params=MessageCreateParamsNonStreaming(...))])`, `.retrieve(id)` (poll `processing_status == "ended"`), `.results(id)` -> streaming `JSONLDecoder[MessageBatchIndividualResponse]`, `.list()`, `.cancel(id)`, `.delete(id)`.

| Fact | Value |
|---|---|
| Discount | 50% on all token usage (stacks with prompt-cache multipliers) |
| Batch limits | ≤ **100,000** requests or **256 MB** per batch, whichever first |
| Timing | most finish < 1 h; hard **24 h** expiry (unfinished -> `expired`, not billed) |
| Results retention | **29 days** after creation (`results_url`) |
| `custom_id` | 1–64 chars matching `^[a-zA-Z0-9_-]{1,64}$`; **results can come back in any order — always join on `custom_id`** |
| Result types | `succeeded` (has `.message`), `errored` (`invalid_request` vs server error; not billed), `canceled`, `expired` |
| Supported | vision, PDFs, tools incl. server tools, system, multi-turn, extended/adaptive thinking, structured outputs, prompt caching, "most beta features" |
| Not supported | `stream:true`, `speed` (fast mode), `max_tokens:0`, `store`/threads, `cache_hint`/`context_hint`, `fallbacks` (item comes back `errored`) |
| Caching in batches | best-effort hits (30–98%); use identical `cache_control` blocks in every request; prefer `ttl:"1h"` since batches can exceed 5 min |
| Extended output | beta `output-300k-2026-03-24` raises `max_tokens` to 300k for Opus 5/4.8/4.7/4.6, Sonnet 5/4.6 (batch only) |
| Spend | batches may slightly overshoot workspace spend limits |

Canopy implication: the N-way "vote/verify" calls per extracted quantity are ideal batch traffic (half price, no RPM pressure); keep the interactive path (progress UI) on synchronous calls with `stream()`.

---

## 8. Structured outputs (`output_config.format`) and strict tools

Source: `build-with-claude/structured-outputs`. GA, no beta header; the old top-level `output_format` request parameter is deprecated in favour of `output_config.format`.

* Request shape: `output_config={"format": {"type": "json_schema", "schema": {...}}}`; response JSON is in the (single) `text` content block. Optionally combine with `output_config.effort`.
* Supported models: `claude-fable-5`, `claude-mythos-5`, `claude-opus-5`, `claude-opus-4-8/4-7/4-6`, `claude-sonnet-5`, `claude-sonnet-4-6`, `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101`, `claude-haiku-4-5-20251001`.
* Mechanism: constrained sampling with a compiled grammar; first use of a schema pays compile latency; grammar cached **24 h from last use**; cache invalidated by schema-structure or tool-set changes (not by `name`/`description` edits). An extra system prompt is injected (slightly more input tokens); changing `output_config.format` invalidates the prompt cache for that thread. Grammar applies only to final output, not thinking or tool calls.
* JSON Schema **supported**: object/array/string/integer/number/boolean/null; `enum` (scalars only), `const`, `anyOf`, `allOf` (not with `$ref`), `$ref`/`$defs`/`definitions` (internal only), `default`, `required`, `additionalProperties:false` (**required on every object**), string `format` in {date-time, time, date, duration, email, hostname, uri, ipv4, ipv6, uuid}, array `minItems` 0 or 1 only.
* **Not supported (400)**: recursive schemas, complex enum members, external `$ref`, `minimum`/`maximum`/`multipleOf`, `minLength`/`maxLength`, other array constraints, `additionalProperties` other than false. `pattern` is not in the supported list [UNVERIFIED whether it 400s or is ignored — the docs only warn not to put PHI in `pattern`].
* Complexity limits per request: ≤ **20** strict tools; ≤ **24** optional (non-`required`) parameters across all strict schemas; ≤ **16** union-typed parameters (`anyOf` or `"type":["string","null"]`); plus internal grammar-size limits ("Schema is too complex for compilation") and a 180 s compile timeout. Tip: make fields required, flatten nesting.
* Enum/const capitalization is not guaranteed — compare case-insensitively.
* Refusal (`stop_reason:"refusal"`) or `max_tokens` cut-off -> output may not match schema; retry with larger `max_tokens`.
* Works with: streaming, batches, token counting, tool use / strict tools together. **Incompatible with citations (400) and prefill.**
* Strict tool use: `"strict": true` on the tool definition (needs `additionalProperties:false` + `required`), guarantees valid `tool_use.input`.
* Python SDK 0.122: `client.messages.parse(model=..., max_tokens=..., messages=..., output_format=MyPydanticModel)` -> `ParsedMessage`; `response.parsed_output` (validated instance). SDK strips unsupported constraints (min/max/minLength/pattern...), adds them to descriptions, adds `additionalProperties:false`, filters formats, then **validates the response client-side against the original Pydantic model** — so we can keep `ge=0`, `le=1` etc. in our models. `parse()` also accepts `output_config`, `thinking`, `tools`, `tool_choice`, `system`, `stream`; note the non-beta `parse()` signature (0.122) does **not** expose top-level `cache_control`/`inference_geo`/`container` — put `cache_control` on content blocks or use `client.beta.messages.parse`, which does. `client.messages.stream(..., output_format=Model)` gives `stream.get_final_message().parsed_output`.

---

## 9. Prompt caching

Source: `build-with-claude/prompt-caching` + pricing.

* Two modes: **automatic** — top-level request field `cache_control={"type":"ephemeral"}` (breakpoint auto-placed on the last cacheable block; uses one of the 4 slots; 400 if 4 explicit breakpoints already exist or last block has a different TTL); **explicit** — `cache_control` on individual blocks, max **4** breakpoints.
* Cacheable blocks: tool definitions, `system` text blocks, `text` blocks in user/assistant turns, **images and documents (user turns)**, `tool_use` and `tool_result` blocks. Not cacheable directly: thinking blocks (cached implicitly when replayed), sub-blocks like citations (cache the parent document block instead).
* Order/hierarchy: `tools` -> `system` -> `messages`; a change invalidates that level and everything after. Tool-definition edits invalidate all; images added/removed invalidate messages; `thinking`/`effort` changes invalidate messages (model-specific for tools/system).
* Minimum cacheable prompt: **512 tokens** (Opus 5, Fable 5, Mythos 5); **1,024** (Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5); **4,096** (Haiku 4.5). Below minimum -> silently uncached (both cache usage fields = 0).
* TTL: 5 min default (`ttl:"5m"`), or `"ttl":"1h"` (2x write). Lifetime measured from the start of the request that writes/reads. 1h entries must precede 5m entries in the prompt.
* Pricing multipliers: write 1.25x (5m) / 2x (1h), read 0.1x. Cache reads don't count toward ITPM.
* Lookback window: each breakpoint looks back at most **20 blocks** for a prior write — add intermediate breakpoints in long agent turns.
* Concurrency: an entry becomes readable only after the first response **begins**; fire the first request, wait for first token, then fan out.
* Isolation: per organization and per workspace on the Claude API.
* Usage fields: `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens`, `usage.input_tokens` (= tokens after last breakpoint).
* Mid-conversation `{"role":"system", ...}` messages (no beta) preserve the cache on Fable 5, Mythos 5, Opus 5, Opus 4.8 and — per the live prompt-caching page — Sonnet 5.
* Pre-warm with `max_tokens: 0` (not in Batches).

Canopy implication: put the immutable protocol + extraction schema + a paper's uploaded PDF (`file_id` document block with `cache_control`) first; keep the per-call task text last. Both extraction and the K verifier calls on the same paper share the cached prefix if issued within 5 min (or use `1h`).

---

## 10. Effort, thinking, refusals, server-side fallbacks

* `output_config.effort`: `low | medium | high | xhigh | max`; default `high`; "effort is a behavioral signal, not a strict token budget"; lower effort also reduces tool calls. On Opus 5, effort does not reliably shorten visible output — prompt for length instead. `max` may over-think on structured-output tasks.
* Thinking: `thinking={"type":"adaptive"}` (+ `"display":"summarized"|"omitted"`); `{"type":"disabled"}` allowed on Sonnet 5 and on Opus 5 only at effort ≤ high; rejected on Fable 5. Thinking blocks stream as `thinking_delta` then one `signature_delta`; pass thinking blocks back unchanged in multi-turn/tool loops. `max_tokens` caps thinking + text together.
* Refusals: HTTP 200 with `stop_reason:"refusal"`, `stop_details:{type:"refusal", category: "cyber"|"bio"|"frontier_llm"|"general_harms"|"reasoning_extraction"|null, explanation}`; pre-output refusal not billed (still counts to rate limits); mid-stream refusal bills streamed tokens. Always check `stop_reason` before reading `content[0]`.
* Server-side fallbacks (beta, Claude API only, not Batches/Bedrock/Vertex/Foundry): request param **`fallbacks`** = `"default"` with header **`server-side-fallback-2026-07-01`** (routes by refusal category), or a list of up to **3** `{model, max_tokens?, thinking?, output_config?, speed?}` entries (works under `server-side-fallback-2026-06-01` too; targets must be in the model's `allowed_fallback_models` from `/v1/models`). Response carries a `fallback` content block per switch and `usage.iterations` entries (`fallback_message` = a fallback model served). Sticky routing ~1 h per org. Python: `client.beta.messages.create(..., fallbacks="default", betas=["server-side-fallback-2026-07-01"])`. Client-side alternative: `Anthropic(middleware=[BetaRefusalFallbackMiddleware([{"model":"claude-opus-4-8"}])])` + `BetaFallbackState`.
* Other stop reasons: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn` (server-tool loop hit its 10-iteration limit; resend the conversation), `model_context_window_exceeded`.

---

## 11. Code execution tool

Source: `agents-and-tools/tool-use/code-execution-tool`. GA, no beta header (legacy beta headers still accepted).

* Tool versions (all GA, all current models incl. Haiku 4.5): `code_execution_20250825` (bash + file ops), `code_execution_20260120` (+ REPL state persistence + programmatic tool calling), `code_execution_20260521` (same runtime; tool description tells Claude about the **90 s wall-clock limit per Python cell** in programmatic tool calling). Declare `{"type":"code_execution_20260521","name":"code_execution"}`. Web search/fetch `_20260209` variants require `_20260120`+.
* Container: Python **3.11**, Linux x86_64, **1 CPU, 5 GiB RAM, 5 GiB disk**, **no internet** (no `pip install`), workspace-scoped, expires **30 days** after creation (checkpointed after ~5 min idle; reuse via top-level `container=<id>` from `response.container.id`).
* Pre-installed: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, pyarrow, openpyxl, xlsxwriter, xlrd, **pillow**, python-pptx, python-docx, pypdf, pdfplumber, pypdfium2, pdf2image, pdfkit, tabula-py, reportlab, img2pdf, sympy, mpmath, tqdm, joblib; CLI: unzip, unrar, 7zip, bc, rg, fd, sqlite. **No OpenCV, no scikit-image, no tesseract listed.**
* Image inputs: yes — upload JPEG/PNG/GIF/WebP (also CSV/XLSX/JSON/XML/text) via Files API and pass `{"type":"container_upload","file_id":...}` (needs `files-api-2025-04-14`); the code can process them with pillow/numpy. Whether Claude can *visually look at* an image it generated inside the container in the same turn is **not documented** — generated files are returned to the client as `file_id`s (`bash_code_execution_output` in the result `content`; captured from `$OUTPUT_DIR`), and "Claude doesn't see the `content` list". PDF `container_upload` is not in the explicit list but PDF libraries are pre-installed [UNVERIFIED].
* Response blocks: `server_tool_use`, `bash_code_execution_tool_result` (`.content.stdout/.stderr/.return_code` or error `error_code` ∈ unavailable, execution_time_exceeded, invalid_tool_input, too_many_requests, output_file_too_large), `text_editor_code_execution_tool_result`; `pause_turn` possible.
* Billing: 1,550 free container-hours/month/org, then $0.05/hour; free with web tools; if files are attached, container time is billed even if the tool isn't called.

Canopy implication: for figure digitization we should run OpenCV locally (our stack) rather than in the sandbox (no OpenCV, no internet); the sandbox is useful only if we want Claude to do ad-hoc numpy/pillow crops with pixel arithmetic and pass results back as files.

---

## 12. Python SDK 0.122 surface (verified against `main` source)

| Call | Signature notes |
|---|---|
| `anthropic.Anthropic(api_key=None, timeout=600s, max_retries=2, base_url=..., middleware=[...])` | Credentials from `ANTHROPIC_API_KEY` / auth token / `ant auth login` profile. Default timeout 10 min. |
| `client.messages.create(model, max_tokens, messages, system=, tools=, tool_choice=, thinking=, output_config=, cache_control=, inference_geo=, container=, metadata=, stop_sequences=, stream=)` | Non-streaming raises `ValueError` ("Streaming is required...") when the estimated time `3600*max_tokens/128000` exceeds 600 s, i.e. **`max_tokens` > 21,333** without `stream=True`/`.stream()`. |
| `client.messages.stream(**same, output_format=Model|dict)` -> `MessageStreamManager` | `with ... as stream:` then `stream.text_stream`, `for event in stream`, `stream.get_final_message()`, `stream.get_final_text()`, `stream.until_done()`, `stream.current_message_snapshot`, `stream.request_id`. Events include `text`, `input_json`, `content_block_stop`, `message_stop`. |
| `client.messages.parse(model, max_tokens, messages, output_format=PydanticType, output_config=, thinking=, tools=, tool_choice=, system=, stream=)` -> `ParsedMessage` | `.parsed_output` (typed) plus normal `.content/.usage/.stop_reason`. `output_format` is merged into `output_config.format`. |
| `client.messages.count_tokens(model, messages, system=, tools=, thinking=)` -> `.input_tokens` | Accepts images/PDFs; free; separate RPM. |
| `client.messages.batches.create(requests=[...])` / `.retrieve(id)` / `.list(limit=)` / `.results(id)` / `.cancel(id)` / `.delete(id)` | `from anthropic.types.messages.batch_create_params import Request`; `from anthropic.types.message_create_params import MessageCreateParamsNonStreaming`. Beta batches: `client.beta.messages.batches.*` (needed for `betas=[...]`). |
| `client.beta.files.upload(file=(name, fileobj, mime) | Path)` / `.list()` / `.retrieve_metadata(id)` / `.delete(id)` / `.download(id)` | Returns `FileMetadata`; download returns `BinaryAPIResponse` with `.write_to_file()`. |
| `client.beta.messages.create(..., betas=["files-api-2025-04-14"], fallbacks=, fallback_credit_token=, speed=, context_management=, diagnostics=, mcp_servers=, user_profile_id=)` | Use for `file_id` sources and any beta flag. `client.beta.messages.parse` also exists (adds `cache_control`, `container`, `betas`). |
| `client.beta.messages.tool_runner(model, max_tokens, messages, tools=[@beta_tool fns | raw server-tool dicts], max_iterations=, stream=, output_format=, compaction_control=, cache_control=, thinking=, output_config=, betas=, ...)` | Iterate for `ParsedBetaMessage`s (or `BetaMessageStream`s with `stream=True`); `runner.until_done()`, `runner.generate_tool_call_response()`, `runner.set_messages_params()`, `runner.append_messages()`. Does not auto-resume `pause_turn`. `from anthropic import beta_tool, beta_async_tool`. |
| `client.models.list()` / `.retrieve(id)` | `max_input_tokens`, `max_tokens`, `capabilities`. |
| Fallback middleware | `from anthropic import BetaRefusalFallbackMiddleware, BetaFallbackState`. |
| Errors | `anthropic.BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError, APIStatusError (.status_code, .type), APIConnectionError, APITimeoutError`; `message._request_id`. |

Beta header cheat sheet: `files-api-2025-04-14` (Files), `server-side-fallback-2026-07-01` / `-2026-06-01` (fallbacks), `output-300k-2026-03-24` (batch 300k output), `compact-2026-01-12` (compaction), `context-management-2025-06-27` (context editing), `task-budgets-2026-03-13`, `fast-mode-2026-02-01`, `code-execution-2025-08-25`+`skills-2025-10-02` (Agent Skills), `mcp-client-2025-11-20`. Structured outputs, effort, adaptive thinking, citations, PDFs, code execution, web search are GA (no header).

---

## 13. Recommendations for Canopy (derived)

1. **Default model split**: Opus 5 (`claude-opus-5`, $5/$25, high-res vision, 512-token cache minimum) for figure digitization and adjudication; Sonnet 5 ($2/$10, high-res vision) for first-pass text/table extraction and the K verifier votes; Haiku 4.5 only for cheap classification (standard-res vision, no effort param, 4,096-token cache minimum). Keep `claude-fable-5` ($10/$50, Start-tier ITPM only 500k) as an optional "escalation" tier.
2. **Ingest via Files API**: upload each PDF once, reference by `file_id` in `document` blocks with `cache_control`; and rasterize pages/figures ourselves with PyMuPDF for anything needing pixel coordinates (server-side PDF rasterization is not mappable).
3. **Provenance without citations in JSON calls**: because citations + `output_config.format` = 400, have the schema carry `page` (1-indexed logical page), `quote` (verbatim), `figure_id`, `bbox_px` fields and verify quotes against PyMuPDF text; use a separate citations-enabled call only where free-text grounding is needed.
4. **Schema discipline**: `additionalProperties:false` everywhere, all fields `required` (use `["number","null"]` sparingly — max 16 union params), no recursion, no numeric bounds in the API schema (Pydantic bounds are enforced client-side by `messages.parse`).
5. **Images**: pre-resize crops to fit 2576 px / 4784 tokens; request absolute pixel bboxes; crop-and-zoom instead of upscaling; keep ≤ 20 image blocks per request or cap dims at 2000 px.
6. **Batching**: route verifier/vote/adjudication traffic through the Batches API (50%, 24 h SLA, join on `custom_id`, `ttl:"1h"` caches); keep the UI-facing extraction synchronous with `messages.stream()` + `get_final_message()`.
7. **Caching layout**: tools -> system(protocol + schema, `cache_control`) -> user(document `file_id` block with `cache_control`) -> task text. Fire the first call per paper, wait for its first token, then fan out.
8. **Always** handle `stop_reason in {"refusal","max_tokens","pause_turn"}`; opt into `fallbacks="default"` on `client.beta.messages` for Opus 5/Fable 5 synchronous calls (not available in batches).
9. Rate-limit design: Start tier gives 2M ITPM/1,000 RPM per model on Opus 5/Sonnet 5; only uncached tokens count, so caching the paper across its ~K+2 calls is the main throughput lever.

---

## 14. Things I could not verify (flagged)

* Whether Batches accept `file_id` document sources when created through the non-beta `client.messages.batches` (docs imply beta features work; use `client.beta.messages.batches` with `betas`).
* Whether `pattern` in a JSON schema is rejected (400) or silently ignored.
* Whether PDFs can be passed as `container_upload` to the code-execution container.
* Whether Claude can visually inspect images produced inside the code-execution container within the same turn.
* Fable 5 30-day-retention requirement is stated in the skill/migration guide, not re-read on a live page today.
* Sonnet 5 support for mid-conversation `role:"system"` messages: live prompt-caching page says yes; the bundled skill says no — trust the live page but test.

---

## Sources (fetched 2026-08-14)

* https://platform.claude.com/docs/en/build-with-claude/pdf-support.md
* https://platform.claude.com/docs/en/build-with-claude/vision.md
* https://platform.claude.com/docs/en/build-with-claude/vision-coordinates.md
* https://platform.claude.com/docs/en/build-with-claude/citations.md
* https://platform.claude.com/docs/en/build-with-claude/files.md
* https://platform.claude.com/docs/en/build-with-claude/batch-processing.md
* https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md
* https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md
* https://platform.claude.com/docs/en/build-with-claude/effort.md
* https://platform.claude.com/docs/en/build-with-claude/thinking.md (adaptive-thinking.md redirects to thinking-steering-and-cost)
* https://platform.claude.com/docs/en/build-with-claude/token-counting.md
* https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons.md
* https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback.md
* https://platform.claude.com/docs/en/about-claude/models/overview.md
* https://platform.claude.com/docs/en/about-claude/pricing.md
* https://platform.claude.com/docs/en/api/rate-limits.md
* https://platform.claude.com/docs/en/api/overview.md (request size limits)
* https://platform.claude.com/docs/en/api/beta-headers.md
* https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool.md
* https://github.com/anthropics/anthropic-sdk-python (README.md, api.md, helpers.md, src/anthropic/resources/messages/messages.py, resources/beta/messages/messages.py, lib/streaming/_messages.py, lib/tools/_beta_runner.py, _constants.py, _base_client.py, __init__.py)
* https://pypi.org/pypi/anthropic/json (0.122.0)
