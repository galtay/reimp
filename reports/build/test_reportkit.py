"""Templates become pages: placeholders filled, TeX rendered, code left alone."""

import pytest
from reportkit import mathml, render


def test_tex_becomes_mathml_outside_code_only() -> None:
    page = mathml(
        r"<p>Let \(x_{ig}\) be a value.</p>"
        r"<div>\[ \mu_g = \frac{1}{n}\sum_i x_{ig} \]</div>"
        r"<script>const re = /\(a\)/; const t = '\(not math\)';</script>"
        r"<style>.a::before { content: '\[x\]'; }</style>"
    )
    assert page.count("<math") == 2
    assert 'display="inline"' in page and 'display="block"' in page
    assert r"'\(not math\)'" in page and r"'\[x\]'" in page
    assert r"\(" not in page.split("<script>")[0]


def test_placeholders_fill_inside_math() -> None:
    page = render({}, {"n": "5000"}, r"<p>\(n = {{n}}\)</p>")
    assert "<mn>5000</mn>" in page


def test_unknown_commands_fail_the_build() -> None:
    with pytest.raises(ValueError, match="does not understand"):
        mathml(r"<p>\(\notacommand x\)</p>")


def test_unbalanced_tex_fails_the_build() -> None:
    with pytest.raises(Exception):  # noqa: B017 — the converter's own error types
        mathml(r"<p>\(\left( x\)</p>")
