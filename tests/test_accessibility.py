"""Static accessibility checks for the UI (WCAG 2.2 AA-oriented).

These run without a browser: they parse ``static/index.html`` and the colour
tokens in ``static/styles.css`` and assert structural and contrast rules.
"""

from __future__ import annotations

import itertools
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"
INTERACTIVE = {"a", "button", "input", "textarea", "select"}


class _Node:
    def __init__(self, tag: str, attrs: dict[str, str | None], parent: _Node | None) -> None:
        self.tag, self.attrs, self.parent = tag, attrs, parent
        self.children: list[_Node] = []
        self.text = ""

    def all_text(self) -> str:
        return (self.text + " ".join(c.all_text() for c in self.children)).strip()

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


class _TreeBuilder(HTMLParser):
    VOID = {"meta", "link", "input", "br", "img", "hr", "source"}

    def __init__(self) -> None:
        super().__init__()
        self.root = _Node("#root", {}, None)
        self.current = self.root
        self.nodes: list[_Node] = []

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, dict(attrs), self.current)
        self.current.children.append(node)
        self.nodes.append(node)
        if tag not in self.VOID:
            self.current = node

    def handle_endtag(self, tag):
        node = self.current
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.current = node.parent

    def handle_data(self, data):
        self.current.text += data


@pytest.fixture(scope="module")
def dom() -> _TreeBuilder:
    builder = _TreeBuilder()
    builder.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return builder


def _by_tag(dom, tag):
    return [n for n in dom.nodes if n.tag == tag]


def _ids(dom):
    return [n.attrs["id"] for n in dom.nodes if n.attrs.get("id")]


# ------------------------------------------------------------------ document


def test_document_language_and_title(dom):
    html = _by_tag(dom, "html")[0]
    assert html.attrs.get("lang") == "en"
    assert _by_tag(dom, "title")[0].all_text()
    viewport = [m for m in _by_tag(dom, "meta") if m.attrs.get("name") == "viewport"]
    assert viewport
    assert "user-scalable=no" not in viewport[0].attrs["content"]  # zoom must stay enabled


def test_landmarks_and_single_h1(dom):
    # header/footer are only page landmarks when not nested in article/section.
    def page_level(node):
        return not any(a.tag in {"article", "section", "main"} for a in node.ancestors())

    for landmark in ("header", "main", "footer"):
        assert len([n for n in _by_tag(dom, landmark) if page_level(n)]) == 1, landmark
    assert len(_by_tag(dom, "h1")) == 1


def test_heading_levels_do_not_skip(dom):
    levels = [int(n.tag[1]) for n in dom.nodes if re.fullmatch(r"h[1-6]", n.tag)]
    for previous, current in itertools.pairwise(levels):
        assert current <= previous + 1, f"h{previous} followed by h{current}"


def test_skip_link_targets_main(dom):
    skip = next(n for n in _by_tag(dom, "a") if "skip-link" in (n.attrs.get("class") or ""))
    target = skip.attrs["href"].lstrip("#")
    assert target in _ids(dom)
    assert dom.nodes.index(skip) < dom.nodes.index(_by_tag(dom, "main")[0])


def test_ids_are_unique(dom):
    ids = _ids(dom)
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------------ forms & controls


def test_every_form_control_has_a_label(dom):
    label_for = {n.attrs.get("for") for n in _by_tag(dom, "label")}
    for control in [n for n in dom.nodes if n.tag in {"input", "textarea", "select"}]:
        wrapped = any(a.tag == "label" for a in control.ancestors())
        named = control.attrs.get("aria-label") or control.attrs.get("aria-labelledby")
        assert control.attrs.get("id") in label_for or wrapped or named, control.attrs


def test_radio_groups_use_fieldset_and_legend(dom):
    for radio in [n for n in _by_tag(dom, "input") if n.attrs.get("type") == "radio"]:
        fieldset = next((a for a in radio.ancestors() if a.tag == "fieldset"), None)
        assert fieldset is not None
        assert any(c.tag == "legend" and c.all_text() for c in fieldset.children)


def test_buttons_have_accessible_names_and_types(dom):
    for button in _by_tag(dom, "button"):
        assert button.all_text() or button.attrs.get("aria-label")
        assert button.attrs.get("type") in {"button", "submit", "reset"}


def test_aria_references_point_to_existing_ids(dom):
    ids = set(_ids(dom))
    for node in dom.nodes:
        for attr in ("aria-describedby", "aria-labelledby", "aria-controls"):
            for ref in (node.attrs.get(attr) or "").split():
                assert ref in ids, f"{attr}={ref} on <{node.tag}>"


def test_no_positive_tabindex(dom):
    for node in dom.nodes:
        if "tabindex" in node.attrs:
            assert int(node.attrs["tabindex"]) <= 0


def test_dynamic_updates_are_announced(dom):
    live = [
        n
        for n in dom.nodes
        if n.attrs.get("aria-live") or n.attrs.get("role") in {"status", "alert"}
    ]
    roles = {n.attrs.get("role") for n in live}
    assert "status" in roles  # agent progress
    assert "alert" in roles  # validation errors


def test_tables_have_header_scope_and_caption(dom):
    for table in _by_tag(dom, "table"):
        headers = [n for n in dom.nodes if n.tag == "th" and table in n.ancestors()]
        assert headers and all(th.attrs.get("scope") for th in headers)
        assert any(n.tag == "caption" for n in table.children)


def test_scrollable_region_is_keyboard_focusable(dom):
    for node in dom.nodes:
        if node.attrs.get("role") == "region":
            assert node.attrs.get("tabindex") == "0"
            assert node.attrs.get("aria-labelledby") or node.attrs.get("aria-label")


# ------------------------------------------------------------------ CSS


CSS = (STATIC / "styles.css").read_text(encoding="utf-8")


def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", block))


def _light_tokens() -> dict[str, str]:
    return _tokens(CSS.split("@media (prefers-color-scheme: dark)")[0])


def _dark_tokens() -> dict[str, str]:
    dark_block = CSS.split("@media (prefers-color-scheme: dark)")[1].split("\n}\n")[0]
    return {**_light_tokens(), **_tokens(dark_block)}


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


TEXT_PAIRS = [
    ("text", "bg"),
    ("text", "surface"),
    ("muted", "surface"),
    ("muted", "bg"),
    ("accent", "surface"),  # links
    ("accent-text", "accent"),  # primary button
    ("ok-fg", "ok-bg"),
    ("warn-fg", "warn-bg"),
    ("bad-fg", "bad-bg"),
    ("na-fg", "na-bg"),
    ("ok-fg", "surface"),  # credibility tier text
    ("warn-fg", "surface"),
    ("bad-fg", "surface"),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize(("fg", "bg"), TEXT_PAIRS)
def test_text_contrast_meets_wcag_aa(theme, fg, bg):
    tokens = _light_tokens() if theme == "light" else _dark_tokens()
    ratio = contrast(tokens[fg], tokens[bg])
    assert ratio >= 4.5, f"{theme}: {fg} on {bg} is {ratio:.2f}:1"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_focus_indicator_contrast(theme):
    tokens = _light_tokens() if theme == "light" else _dark_tokens()
    assert contrast(tokens["focus"], tokens["surface"]) >= 3.0  # WCAG 2.4.13 / 1.4.11


def test_css_supports_reduced_motion_and_visible_focus():
    assert "prefers-reduced-motion" in CSS
    assert ":focus-visible" in CSS
    assert "outline: none" not in CSS


def test_contrast_helper_known_values():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#777777") == pytest.approx(1.0)
