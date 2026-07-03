import pytest

from kernelthing import gates

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
    items = [
        {
            "file": "k.py",
            "classification": "hacked",
            "reason": "high_critical",
            "patterns": ["TIMER_MONKEYPATCH"],
        }
    ]
    out = gates.format_kernelguard_violations(items)
    assert "k.py [hacked]: high_critical" in out
    assert "TIMER_MONKEYPATCH" in out
