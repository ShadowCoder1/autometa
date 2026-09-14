"""Orthographic and suffix variants of a search term, and the rendering of a block into a string.

The model writes the CONCEPTS (`blocks.py`); this module writes the FORMS. A paper that says
`non-linear` is not found by `nonlinear` on an index that does not stem, and Europe PMC does not
stem: the miss diagnosis (design 01) found a noun failing to reach its adjective, a plural failing
to reach its singular, and a quoted hyphenated form failing to reach the closed one. So every
term gets up to three variants — hyphen / space / closed forms for a compound, a suffix-family
sibling (`tolerant` ↔ `tolerance`), a plural or singular — listed under `expanded` in the plan so
a reader can see the derivation, and never more than three, because the adversarial review
measured an unbounded expander producing `fource field`, `sensourimotour`, `adultss` and
`adaptates` and then dropping the model's own `intermanual transfer` to make room for them
(design 04 BLOCKER-2c, m6).

Three rules that are NOT here, on purpose: no `ation → ate` (it makes `adaptate`), no substring
`our → or` (it makes `fource`), and no plural of a plural (`adultss`). A prefix is only split off
(`aftereffect → after-effect`) when what is left is a word the model or the protocol actually
used, so `compensation` never becomes `co-mpensation` and `contrast` never `contra-st` (04 v2
verification, edit 2).

Rendering: a term containing a space OR a hyphen is quoted. Europe PMC splits a bare hyphenated
token into two AND'd words — `after-effect` bare matched 7.4 million records and quoted 4,360
(04 BLOCKER-2a) — and the same string is sent to every index, so the rule is the strict one.

Nothing here knows a research field: the prefixes are English morphology, the suffix pairs are
English morphology, and every word that goes through them came from the user's own question,
protocol or the model's reading of them.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

__all__ = ["MAX_VARIANTS", "MAX_STRING_CHARS", "MIN_HEAD_WORD", "PREFIXES", "STOPWORDS",
           "variants", "head_words", "render_term", "render_block", "render_query", "singular",
           "term_tokens", "known_words"]

#: variants per term. Three: the hyphen/space/closed triple uses all of them for a compound, and
#: the review found every form past the third was a non-word.
MAX_VARIANTS = 3

#: the longest string OpenAlex's `search=` accepts is 1,500 characters (HTTP 400 above it,
#: measured); the full-text form is emitted only under this, with a margin for the filter syntax.
MAX_STRING_CHARS = 1400

#: a bare head or modifier word shorter than this is not worth a slot (`task`, `hand`, `arm`)
MIN_HEAD_WORD = 5

#: words that carry no search signal on their own. Deliberately short and general — a stopword
#: list that grew domain terms would be this module quietly deciding what a field is about.
STOPWORDS: frozenset[str] = frozenset("""
a an the and or of in on for to from with without by is are was were be been being do does did
this that these those it its as at than then there their they them we our us you your i
does effect effects study studies research compared comparison versus vs between among any
what which who whom how why when where whether more most less least much many differently
different once removed other others second first
""".split())

#: prefixes a closed compound may be split at (`aftereffect → after-effect`), when the remainder
#: is itself a word the protocol or the model used. English morphology, not a field's vocabulary.
PREFIXES: tuple[str, ...] = ("non", "inter", "intra", "after", "pre", "post", "co", "visuo",
                             "sensori", "contra")

#: British → American, whole-suffix on the head only (never a substring swap)
_BRITISH: tuple[tuple[str, str], ...] = (("isation", "ization"), ("ise", "ize"), ("our", "or"),
                                         ("aemia", "emia"), ("tre", "ter"))

#: suffix families, on the SINGULAR head, when the head is long enough that the suffix is a
#: suffix and not most of the word. No `ation → ate`.
_SUFFIX: tuple[tuple[str, str], ...] = (("ant", "ance"), ("ance", "ant"), ("ent", "ence"),
                                        ("ence", "ent"), ("ate", "ation"), ("ization", "ize"),
                                        ("ize", "ization"), ("ized", "ize"))

_WORD = re.compile(r"[a-z][a-z\-']*[a-z]|[a-z]")


def term_tokens(term: str) -> list[str]:
    """The words of a term, lower-cased, hyphens split — the unit everything below reasons in."""
    return [t for t in re.split(r"[\s\-]+", " ".join(str(term).lower().split())) if t]


def known_words(*sources: Iterable[str] | str) -> frozenset[str]:
    """Every word in the model's own terms and the protocol's text: the boundary the prefix rule
    checks against, so a split only ever produces a word somebody actually wrote."""
    words: set[str] = set()
    for source in sources:
        texts = [source] if isinstance(source, str) else list(source)
        for text in texts:
            words.update(_WORD.findall(str(text).lower().replace("-", " ")))
    return frozenset(words)


def singular(head: str) -> str:
    """`dynamics → dynamic`, `hands → hand`, `studies → study`; a word that is not a plural comes
    back unchanged (`analysis`, `class`, `canvas`, `bias`)."""
    if len(head) <= 3 or head.endswith(("ss", "us", "is", "ous")):
        return head
    if head.endswith("ies") and len(head) > 4:
        return head[:-3] + "y"
    if head.endswith(("sses", "xes", "ches", "shes")):
        return head[:-2]
    if head.endswith("s"):
        return head[:-1]
    return head


def _plural(head: str) -> str:
    if head.endswith("y") and len(head) > 1 and head[-2] not in "aeiou":
        return head[:-1] + "ies"
    if head.endswith(("s", "x", "ch", "sh")):
        return head + "es"
    return head + "s"


def variants(term: str, known: Iterable[str] | frozenset[str] = frozenset()) -> list[str]:
    """Up to `MAX_VARIANTS` other forms of `term`, in a fixed order, never the term itself.

    Order: compound forms (hyphen, space, closed) or the prefix-hyphen form; British→American on
    the head; a suffix-family sibling of the singular head; the plural of a singular head or the
    singular of a plural one. The first three win, so a compound's spelling forms outrank its
    plural — the review measured the spelling forms as the ones that lose papers.
    """
    text = " ".join(str(term).lower().split())
    if not text:
        return []
    known_set = known if isinstance(known, frozenset) else frozenset(known)
    out: list[str] = []

    def add(candidate: str) -> None:
        candidate = " ".join(candidate.split())
        if candidate and candidate != text and candidate not in out and len(out) < MAX_VARIANTS:
            out.append(candidate)

    tokens = term_tokens(text)
    head = tokens[-1]
    stem = text[: len(text) - len(head)] if text.endswith(head) else ""
    if "-" in text:
        # a hyphenated term: its space form and its closed form (the hyphen form is the term)
        add(text.replace("-", " "))
        add(text.replace("-", ""))
    elif len(tokens) == 2:
        add("-".join(tokens))
        add("".join(tokens))
    elif len(tokens) == 1:
        for prefix in PREFIXES:
            rest = head[len(prefix):]
            if (head.startswith(prefix) and len(rest) >= 4 and rest in known_set
                    and rest != head):
                add(f"{prefix}-{rest}")
                break

    for british, american in _BRITISH:
        if head.endswith(british) and len(head) > len(british) + 2:
            add(stem + head[: -len(british)] + american)
            break

    base = singular(head)
    for suffix, sibling in _SUFFIX:
        if base.endswith(suffix) and len(base) > len(suffix) + 3:
            add(stem + base[: -len(suffix)] + sibling)
            break

    if base == head:
        # a participle (`preferred`, `right-handed`) and a Greek singular (`analysis`) have no
        # `+s` plural worth a slot; everything else gets its plural
        if not head.endswith(("ed", "is")):
            add(stem + _plural(head))
    else:
        add(stem + base)
    return out[:MAX_VARIANTS]


def head_words(term: str) -> list[str]:
    """The bare head (last token) and modifier (first token) of a MULTI-WORD term, each only when
    it is at least `MIN_HEAD_WORD` letters and not a stopword — `"motor adaptation"` →
    `["adaptation", "motor"]`. Design 03 §1, v3 amendment: a quoted phrase matches only that
    phrase, and the step-3 measurement lost four answer-key papers to blocks made of phrases; the
    bare words reach them, and width control measures and prunes the ones that are too wide.
    A single-word term yields nothing: it is already bare."""
    text = " ".join(str(term).lower().split())
    tokens = text.split()                       # a hyphenated compound is one word here
    if len(tokens) < 2:
        return []
    out: list[str] = []
    for word in (tokens[-1], tokens[0]):
        if (len(word) >= MIN_HEAD_WORD and word not in STOPWORDS and word.isalpha()
                and word not in out):
            out.append(word)
    return out


def render_term(term: str) -> str:
    """Quoted when it holds a space or a hyphen; bare otherwise. Internal quotes are dropped."""
    clean = " ".join(str(term).replace('"', " ").split())
    return f'"{clean}"' if (" " in clean or "-" in clean) else clean


def render_block(terms: Sequence[str]) -> str:
    """`(t1 OR "t2 t3" OR "t4-t5")` — one concept, its forms OR'd."""
    rendered = [render_term(t) for t in terms if str(t).strip()]
    return "(" + " OR ".join(dict.fromkeys(rendered)) + ")" if rendered else ""


def render_query(blocks: Sequence[Sequence[str]]) -> str:
    """The blocks AND'd; an empty block contributes nothing rather than `()`."""
    return " AND ".join(part for part in (render_block(b) for b in blocks) if part)
