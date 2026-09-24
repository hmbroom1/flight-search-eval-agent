"""Guards for the report template.

A renderer that is called but no longer defined throws at load time and takes
every section after it with it, which is how the run timeline, the charts and
the before/after section once went missing from a published report. These are
cheap structural checks, not a substitute for looking at the page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "scripts" / "report_template.html"


@pytest.fixture(scope="module")
def template() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def defined_functions(src: str) -> set[str]:
    return set(re.findall(r"function\s+([A-Za-z_]\w*)\s*\(", src))


def test_every_called_renderer_is_defined(template):
    """The init list drives the page; a name in it that isn't defined breaks the render."""
    init = re.search(r"\[(render[^\]]+)\]\s*\n?\s*\.forEach", template)
    assert init, "expected the init list of renderers"
    called = [n.strip() for n in init.group(1).split(",")]
    missing = sorted(set(called) - defined_functions(template))
    assert not missing, f"renderers called but not defined: {missing}"


def test_renderers_target_ids_that_exist(template):
    """getElementById targets in the script must exist in the markup."""
    markup = template.split("<script>")[0]
    ids = set(re.findall(r'id="([^"]+)"', markup))
    used = set(re.findall(r'getElementById\("([^"]+)"\)', template))
    # about-models is filled by renderHero and lives inside the hero copy.
    missing = sorted(used - ids)
    assert not missing, f"script targets ids not present in the markup: {missing}"


def test_no_orphaned_render_helpers(template):
    """A renderer nothing calls is dead code, usually left behind by an edit."""
    defined = {n for n in defined_functions(template) if n.startswith("render")}
    init = re.search(r"\[(render[^\]]+)\]\s*\n?\s*\.forEach", template)
    listed = {n.strip() for n in init.group(1).split(",")} if init else set()
    orphans = sorted(
        n for n in defined
        if n not in listed and len(re.findall(rf"\b{n}\s*\(", template)) < 2
    )
    assert not orphans, f"defined but never called: {orphans}"


def test_model_palette_covers_every_model_slot(template):
    """mcol()/tint() index into --m0..--mN; the palette must define that many colours."""
    mod = re.search(r"var\(--m\$\{i % (\d+)\}\)", template)
    assert mod, "expected mcol() to index the model palette"
    slots = int(mod.group(1))
    for theme_block in re.findall(r"--m0:[^;]+;(?:\s*--m\d+:[^;]+;)*", template):
        defined = set(re.findall(r"--m(\d+):", theme_block))
        assert {str(i) for i in range(slots)} <= defined, (
            f"palette defines {sorted(defined)} but mcol() uses {slots} slots"
        )
