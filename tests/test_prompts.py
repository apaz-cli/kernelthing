from kernelthing import prompts
from kernelthing.config import PROMPTS_DIR


def test_single_pass_no_rescan():
    # A value containing a placeholder must NOT be re-expanded.
    out = prompts.render("{{A}} {{B}}", A="{{B}}", B="value")
    assert out == "{{B}} value"


def test_missing_var_left_intact():
    assert prompts.render("x {{UNKNOWN}} y", FOO="bar") == "x {{UNKNOWN}} y"


def test_basic_substitution():
    assert prompts.render("round {{N}}", N=3) == "round 3"


def test_load_real_prompt_renders_placeholders():
    # regular-review.md should exist in the working set and accept CURRENT_ROUND.
    text = prompts.load("codex/regular-review.md")
    assert text, "regular-review.md should be present"
    rendered = prompts.render(text, CURRENT_ROUND=7)
    assert "Round 7" in rendered
    assert "{{CURRENT_ROUND}}" not in rendered


def test_working_set_has_no_role_word_claude():
    # The mechanical rename should have removed the literal agent word "Claude".
    bad = []
    for p in PROMPTS_DIR.rglob("*.md"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "Claude" in line:
                bad.append(f"{p.name}:{i}")
    assert not bad, f"unexpected 'Claude' occurrences: {bad}"


def test_load_and_render_safe_fallback():
    out = prompts.load_and_render_safe("does/not/exist.md", "fallback {{X}}", X="ok")
    assert out == "fallback ok"
