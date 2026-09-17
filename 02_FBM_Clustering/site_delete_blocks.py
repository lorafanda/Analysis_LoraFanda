#!/usr/bin/env python3
"""
site_delete_blocks.py - delete from analysis_status.html the blocks make_site_ui.py had
struck through as proposed deletions (its DEL table), for real.

    python site_delete_blocks.py --check     say what would go, change nothing
    python site_delete_blocks.py --apply     delete, then clear DEL in make_site_ui.py

Same addressing as the runtime layer: (section id, group key, text prefix or None). A group
is a top-level <h3> of the section - keyed by its id, else the slug of its text - and
everything up to the next top-level <h3>; "intro" is the section's top level before the
first <h3>. A prefix names the first top-level element of that group whose text
(whitespace removed, case-insensitive) starts with it. Nothing inside a generator's
BEGIN/END block is touched, and a deletion that would cross a marker is refused.
Elements are located with a small tag-balancing scanner over the raw text, so the rest of
the file is left byte for byte as it was; the removed HTML is written to a .removed file
beside this script, one block per entry, for the record (git has it as well).
"""
from __future__ import annotations

import argparse
import html as htmlmod
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_site_ui as ui  # noqa: E402  (DEL, SITE)

TAG = re.compile(r"<!--.*?-->|<(/?)([a-zA-Z][a-zA-Z0-9-]*)(?:\s[^>]*)?>", re.S)
VOID = {"img", "br", "hr", "input", "meta", "link", "source", "wbr", "col"}


def slug(t: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return t[:48] or "x"


def text_of(fragment: str) -> str:
    frag = re.sub(r"<!--.*?-->", "", fragment, flags=re.S)
    frag = re.sub(r"<[^>]+>", " ", frag)
    return re.sub(r"\s+", " ", htmlmod.unescape(frag)).strip()


def squash(t: str) -> str:
    return re.sub(r"\s+", "", t).lower()


def top_level(s: str, start: int, end: int):
    """(start, end, tag) of every depth-0 element in s[start:end]; comments skipped."""
    depth, cur, out = 0, None, []
    for m in TAG.finditer(s, start, end):
        if m.group(0).startswith("<!--"):
            continue
        closing, tag = m.group(1) == "/", m.group(2).lower()
        if tag in VOID:
            if depth == 0:
                out.append((m.start(), m.end(), tag))
            continue
        if not closing:
            if depth == 0:
                cur = (m.start(), tag)
            depth += 1
        else:
            depth -= 1
            if depth == 0 and cur:
                out.append((cur[0], m.end(), cur[1])); cur = None
    return out


def section_span(s: str, sid: str):
    i = s.index(f'<section id="{sid}"')
    els = top_level(s, i, len(s))
    assert els and els[0][2] == "section", sid
    return i, els[0][1]


def groups_of(s: str, sid: str):
    """key -> (start, end) of each top-level h3 group, plus 'intro' -> the top level before the first h3."""
    a, b = section_span(s, sid)
    inner = top_level(s, s.index(">", a) + 1, b - len("</section>"))
    h3s = [e for e in inner if e[2] == "h3"]
    out = {}
    for k, (hs, he, _) in enumerate(h3s):
        head = s[hs:he]
        m = re.search(r'id="([^"]+)"', head[:head.index(">")])
        key = m.group(1) if m else slug(text_of(head))
        nxt = h3s[k + 1][0] if k + 1 < len(h3s) else b - len("</section>")
        out[key] = (hs, nxt)
    first_h3 = h3s[0][0] if h3s else b - len("</section>")
    out["intro"] = (s.index(">", a) + 1, first_h3)
    return out


def find_target(s: str, sid: str, key: str, prefix: str | None):
    groups = groups_of(s, sid)
    if key not in groups:
        raise KeyError(f"{sid}: no group {key!r}; have {sorted(groups)[:20]}")
    gs, ge = groups[key]
    if prefix is None:
        return gs, ge
    want = squash(prefix)
    for es, ee, tag in top_level(s, gs, ge):
        if tag == "h3" or tag == "h2":
            continue
        if squash(text_of(s[es:ee])).startswith(want):
            return es, ee
    raise KeyError(f"{sid}/{key}: no element starting with {prefix!r}")


def widen(s: str, a: int, b: int):
    """Take the whitespace before the block (to the line start) and the newline after it."""
    la = s.rfind("\n", 0, a) + 1
    if s[la:a].strip() == "":
        a = la
    if s[b:b + 1] == "\n":
        b += 1
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not ui.DEL:
        print("DEL is empty - nothing to delete"); return 0
    s = ui.SITE.read_text(encoding="utf-8")
    cuts = []
    HDR = re.compile(r'^<(h2\b|div class="eyebrow"|p class="lead"|div class="callout retired-note")')
    for sec, key, prefix, why in ui.DEL:
        if key == "intro" and prefix is None:
            # the whole intro = every top-level element before the first heading that is not the tab's own header
            gs, ge = groups_of(s, sec)["intro"]
            spans = [(es, ee) for es, ee, tag in top_level(s, gs, ge) if not HDR.match(s[es:ee])]
        else:
            spans = [find_target(s, sec, key, prefix)]
        for st, en in spans:
            st, en = widen(s, st, en)
            frag = s[st:en]
            if "<!-- BEGIN" in frag or "<!-- END" in frag:
                raise SystemExit(f"{sec}/{key}: the block spans a generator marker - refused")
            cuts.append((st, en, sec, key, prefix, why, frag))
    # consecutive whole groups overlap by the indentation of the next heading; those merge
    cuts.sort()
    for (a1, b1, *_), (a2, b2, *_) in zip(cuts, cuts[1:]):
        if b1 > a2:
            assert s[a2:b1].strip() == "", "overlapping cuts with content between"
    words = sum(len(text_of(c[6]).split()) for c in cuts)
    print(f"{len(cuts)} blocks, {sum(b - a for a, b, *_ in cuts):,} chars, ~{words:,} words")
    for st, en, sec, key, prefix, why, frag in cuts:
        print(f"  {sec:9s} {key[:32]:32s} {('[' + prefix[:28] + ']') if prefix else '[whole group]':32s} {len(text_of(frag).split()):5d} w")
    if a.check:
        print("(--check: nothing written)"); return 0
    removed = ["<!-- removed from analysis_status.html on 2026-09-17 by site_delete_blocks.py -->"]
    done_to = len(s) + 1
    for st, en, sec, key, prefix, why, frag in reversed(cuts):
        removed.append(f"\n<!-- {sec} / {key} / {prefix or 'whole group'} : {why} -->\n{frag}")
        en = min(en, done_to)          # the overlap with the cut already made
        s = s[:st] + s[en:]
        done_to = st
    ui.SITE.write_text(s, encoding="utf-8")
    (HERE / "site_delete_blocks.removed.txt").write_text("\n".join(reversed(removed)) + "\n", encoding="utf-8")
    # the table is done its job: clear it, keeping the record in git
    src = (HERE / "make_site_ui.py").read_text(encoding="utf-8")
    i = src.index("DEL = [")
    j = src.index("\n]\n", i) + 3
    note = ("DEL = [\n    # the 41 blocks proposed on 2026-09-17 were deleted for real by site_delete_blocks.py the same\n"
            "    # day (their HTML is in site_delete_blocks.removed.txt and in git); add entries here to propose more\n]\n")
    (HERE / "make_site_ui.py").write_text(src[:i] + note + src[j:], encoding="utf-8")
    print(f"wrote {ui.SITE}; DEL cleared in make_site_ui.py; removed HTML in site_delete_blocks.removed.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
