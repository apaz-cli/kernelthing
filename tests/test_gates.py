import pytest

from kernelthing import gates
from kernelthing.config import (
    VERDICT_ADVANCED,
    VERDICT_REGRESSED,
    VERDICT_STALLED,
    VERDICT_UNKNOWN,
)


def test_has_complete_exact_last_line():
    assert gates.has_complete("some review\n\nCOMPLETE\n")
    assert not gates.has_complete("COMPLETE\nmore text after")
    assert not gates.has_complete("almost COMPLETE here")


def test_has_stop():
    assert gates.has_stop("findings...\nSTOP")
    assert not gates.has_stop("STOP\ntrailing")


def test_parse_verdict():
    assert gates.parse_verdict("Mainline Progress Verdict: ADVANCED") == VERDICT_ADVANCED
    assert gates.parse_verdict("x\nMainline Progress Verdict: STALLED\ny") == VERDICT_STALLED
    assert gates.parse_verdict("Mainline Progress Verdict: REGRESSED") == VERDICT_REGRESSED
    assert gates.parse_verdict("no verdict here") == VERDICT_UNKNOWN


def test_parse_verdict_takes_last():
    text = "Mainline Progress Verdict: STALLED\n...\nMainline Progress Verdict: ADVANCED"
    assert gates.parse_verdict(text) == VERDICT_ADVANCED


def test_scan_p_markers():
    text = "issue one [P0]\nfoo [P2] bar\nnothing\n[P0] again"
    assert gates.scan_p_markers(text) == ["[P0]", "[P2]"]
    assert gates.scan_p_markers("clean review\nCOMPLETE") == []


def test_bitlesson_none_ok():
    summary = "## BitLesson Delta\n- Action: none\n- Lesson ID(s): NONE\n- Notes: n/a"
    assert gates.bitlesson_delta_error(summary) is None


def test_bitlesson_missing_section():
    err = gates.bitlesson_delta_error("no delta here")
    assert err and err.template == "bitlesson-delta-missing"


def test_bitlesson_add_requires_valid_id():
    # malformed token (illegal char) is rejected
    bad = "## BitLesson Delta\n- Action: add\n- Lesson ID(s): bad!id\n- Notes: did a thing"
    assert gates.bitlesson_delta_error(bad) is not None
    # both the strict BL- form and a natural slug are accepted
    for lid in ("BL-20260604-tiling", "round-0-register-blocking"):
        good = (f"## BitLesson Delta\n- Action: add\n- Lesson ID(s): {lid}\n"
                "- Notes: shared-memory tiling fixed bank conflicts")
        assert gates.bitlesson_delta_error(good) is None


def test_bitlesson_add_requires_notes():
    no_notes = "## BitLesson Delta\n- Action: add\n- Lesson ID(s): round-0-x\n- Notes: "
    err = gates.bitlesson_delta_error(no_notes)
    assert err and err.template == "bitlesson-delta-missing-notes" and err.action == "add"


def test_bitlesson_none_with_ids_inconsistent():
    bad = "## BitLesson Delta\n- Action: none\n- Lesson ID(s): BL-20260604-x\n- Notes: x"
    err = gates.bitlesson_delta_error(bad)
    assert err and err.template == "bitlesson-delta-inconsistent"


# --- goal tracker initialization ---

def test_goal_tracker_uninitialized_detects_placeholders(tmp_path):
    gt = tmp_path / "goal-tracker.md"
    gt.write_text(
        "## IMMUTABLE SECTION\n\n### Ultimate Goal\n[To be filled in Round 0 from the plan]\n\n"
        "### Acceptance Criteria\n[To be filled in Round 0 from the plan]\n\n"
        "## MUTABLE SECTION\n\n#### Active Tasks\n", encoding="utf-8")
    missing = gates.goal_tracker_uninitialized(gt)
    assert "Ultimate Goal" in missing and "Acceptance Criteria" in missing


def test_goal_tracker_initialized_ok(tmp_path):
    gt = tmp_path / "goal-tracker.md"
    gt.write_text(
        "## IMMUTABLE SECTION\n\n### Ultimate Goal\nMaximize TFLOPS\n\n"
        "### Acceptance Criteria\n- correct=true\n\n"
        "## MUTABLE SECTION\n\n#### Active Tasks\n- tile the matmul\n", encoding="utf-8")
    assert gates.goal_tracker_uninitialized(gt) == ""


# --- bitlesson empty-KB ---

def test_bitlesson_empty_kb_rejects_none():
    summary = "## BitLesson Delta\n- Action: none\n- Lesson ID(s): NONE\n"
    seed = "# BitLessons\n\nProject-specific, hard-won lessons.\n"
    err = gates.bitlesson_delta_error(summary, kb_text=seed)
    assert err and err.template == "bitlesson-delta-empty-kb"


def test_bitlesson_none_ok_when_kb_populated():
    summary = "## BitLesson Delta\n- Action: none\n- Lesson ID(s): NONE\n"
    kb = "# BitLessons\n\n- BL-20260101-tiling: use 128x128 tiles\n"
    assert gates.bitlesson_delta_error(summary, kb_text=kb) is None


def test_bitlesson_none_ok_when_no_kb_supplied():
    summary = "## BitLesson Delta\n- Action: none\n- Lesson ID(s): NONE\n"
    assert gates.bitlesson_delta_error(summary) is None


# --- git-not-clean notes ---

def test_git_not_clean_note_templates():
    assert gates.git_not_clean_note_templates("?? build/out.o\n M src/x.c") == ["git-not-clean-untracked"]
    assert gates.git_not_clean_note_templates(" M .humanize/rlcr/x/state.json") == ["git-not-clean-humanize-local"]
    assert gates.git_not_clean_note_templates(" M src/x.c") == []


# --- incomplete-todo gating (port of check-todos-from-transcript) ---

def test_classify_lane():
    assert gates.classify_lane("[mainline] tile the matmul") == "mainline"
    assert gates.classify_lane("[BLOCKING] fix the build") == "blocking"
    assert gates.classify_lane("[queued] cleanup later") == "queued"
    # No leading tag -> defaults to blocking (safety).
    assert gates.classify_lane("just do the thing") == "blocking"
    # An inline mention must NOT downgrade an otherwise-blocking task.
    assert gates.classify_lane("fix docs that mention [queued] work") == "blocking"


def _write_todos(tmp_path, session, todos):
    """Seed opencode's SQLite todo store (schema mirrors opencode 1.16's `todo`)."""
    import sqlite3
    db = gates.opencode_db_file(data_home=tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE IF NOT EXISTS todo (session_id text, content text, status text, "
        "priority text, position integer, time_created integer, time_updated integer)")
    for pos, t in enumerate(todos):
        con.execute(
            "INSERT INTO todo (session_id, content, status, priority, position, "
            "time_created, time_updated) VALUES (?, ?, ?, 'medium', ?, 0, 0)",
            (session, t["content"], t["status"], pos))
    con.commit()
    con.close()
    return db


def test_incomplete_todos_all_completed(tmp_path):
    _write_todos(tmp_path, "ses_1", [
        {"content": "[mainline] a", "status": "completed"},
        {"content": "[blocking] b", "status": "cancelled"},
    ])
    assert gates.incomplete_todos("ses_1", data_home=tmp_path) == []


def test_incomplete_todos_blocks_open_mainline_and_blocking(tmp_path):
    _write_todos(tmp_path, "ses_2", [
        {"content": "[mainline] a", "status": "in_progress"},
        {"content": "[blocking] b", "status": "pending"},
        {"content": "untagged c", "status": "pending"},
    ])
    items = gates.incomplete_todos("ses_2", data_home=tmp_path)
    lanes = sorted(i["lane"] for i in items)
    assert lanes == ["blocking", "blocking", "mainline"]


def test_incomplete_todos_queued_never_blocks(tmp_path):
    _write_todos(tmp_path, "ses_3", [
        {"content": "[queued] follow-up", "status": "pending"},
        {"content": "[mainline] done", "status": "completed"},
    ])
    assert gates.incomplete_todos("ses_3", data_home=tmp_path) == []


def test_incomplete_todos_missing_store_fails_open(tmp_path):
    assert gates.incomplete_todos("ses_missing", data_home=tmp_path) == []
    assert gates.incomplete_todos(None, data_home=tmp_path) == []


def test_format_incomplete_todos():
    items = [
        {"status": "pending", "content": "[mainline] a", "lane": "mainline"},
        {"status": "in_progress", "content": "untagged b", "lane": "blocking"},
    ]
    out = gates.format_incomplete_todos(items)
    assert out == "  - [pending] [mainline] [mainline] a\n  - [in_progress] [blocking] untagged b"


# --- kernelguard cheat detection ---

# A blatant timer-monkeypatch cheat: kernelguard's TIMER_MONKEYPATCH rule fires.
_CHEAT = "import torch\ntorch.cuda.Event = lambda *a, **k: None\n"
# An honest kernel snippet with no gaming patterns.
_CLEAN = "import torch\ndef custom_kernel(a, b):\n    return a @ b\n"


def test_kernelguard_flags_cheat(tmp_path):
    pytest.importorskip("kernelguard")
    (tmp_path / "k.py").write_text(_CHEAT, encoding="utf-8")
    out = gates.kernelguard_violations(["k.py"], tmp_path)
    assert [v["file"] for v in out] == ["k.py"]
    assert out[0]["patterns"]  # at least one named rule


def test_kernelguard_allows_clean(tmp_path):
    pytest.importorskip("kernelguard")
    (tmp_path / "k.py").write_text(_CLEAN, encoding="utf-8")
    assert gates.kernelguard_violations(["k.py"], tmp_path) == []


def test_kernelguard_missing_file_fails_open(tmp_path):
    # No file written: the gate must not raise, just report nothing.
    assert gates.kernelguard_violations(["nope.py"], tmp_path) == []


def test_format_kernelguard_violations():
    items = [{"file": "k.py", "classification": "hacked",
              "reason": "high_critical", "patterns": ["TIMER_MONKEYPATCH"]}]
    out = gates.format_kernelguard_violations(items)
    assert "k.py [hacked]: high_critical" in out
    assert "TIMER_MONKEYPATCH" in out
