"""Comparison reports keep missing observations and terminal support explicit."""

import pytest
from remax.results import compare, comparison_markdown


def test_terminal_support_is_not_paired_baseline_support():
    report = compare(domain="countdown")
    row = next(r for r in report["rows"] if r["method"] == "replay_maxrl")
    assert row["pcmd_seeds"] == [43, 44, 45, 46, 47]
    assert row["pass8"] == pytest.approx((83 + 86 + 86 + 86 + 85) / (128 * 5))
    assert row["pcmd"] == pytest.approx(0.5108559997942338)
    assert report["min_defined_prompts"] == 30
    assert not report["numerical_reproduction_performed"]


def test_insufficient_diversity_support_is_not_zero_or_filled_from_other_domains():
    report = compare(domain="python_factors")
    missing = [r for r in report["rows"] if r["pcmd"] is None]
    assert missing
    assert all(r["pcmd_seeds"] == [] for r in missing)
    assert "—" in comparison_markdown(report)


def test_level_two_does_not_invent_a_grpo_comparator():
    report = compare("level2", "qwen05b")
    assert report["methods"] == ["drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl"]


@pytest.mark.parametrize(
    "level,scale,domain",
    [
        ("level3", "qwen05b", None),
        ("level1", "unknown", None),
        ("level1", "qwen05b", "unknown"),
    ],
)
def test_unavailable_comparisons_fail_explicitly(level, scale, domain):
    with pytest.raises(ValueError, match="no saved-key comparison"):
        compare(level, scale, domain)


def test_excluded_seed_stays_absent_from_terminal_comparison():
    report = compare("level2", "qwen05b", "pantry_plan")
    row = next(r for r in report["rows"] if r["method"] == "replay_maxrl")
    assert row["terminal_seeds"] == [43, 44, 45, 47]
    assert 46 not in row["pcmd_seeds"]
