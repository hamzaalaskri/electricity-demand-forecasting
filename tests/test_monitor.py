import numpy as np
import pytest

from src.monitor import (
    performance_alert,
    population_stability_index,
    promote,
    quality_issues,
    should_roll_back,
)


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 20_000)
    assert population_stability_index(reference, rng.normal(0, 1, 5_000)) < 0.02


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 20_000)
    assert population_stability_index(reference, rng.normal(1.5, 1, 5_000)) > 0.25


def test_clean_readings_have_no_issues():
    readings = 30_000 + 3_000 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    assert quality_issues(readings) == []


@pytest.mark.parametrize(
    "readings, expected",
    [
        ([30_000.0] * 20 + [np.nan] * 4, "4 missing readings"),
        ([30_000.0, 31_000.0, 0.0, 30_500.0], "non-positive readings"),
        ([30_000.0, 31_000.0] + [29_000.0] * 8, "flat-lined for 8 hours"),
        ([30_000.0, 31_000.0, 60_000.0, 31_000.0], "implausible hour-to-hour jump"),
    ],
)
def test_quality_problems_are_reported(readings, expected):
    assert expected in quality_issues(np.array(readings))


def test_performance_alert_needs_consecutive_days():
    reference = 0.6
    bad = reference * 1.3
    assert performance_alert([bad] * 7, reference)
    assert not performance_alert([bad] * 6, reference)
    assert not performance_alert([bad] * 6 + [reference], reference)


def test_promotion_gate_requires_minimum_improvement():
    assert promote(candidate_mae=980.0, champion_mae=1000.0)
    assert not promote(candidate_mae=995.0, champion_mae=1000.0)
    assert not promote(candidate_mae=1100.0, champion_mae=1000.0)


def test_rollback_tolerance():
    assert should_roll_back(champion_mae=1100.0, previous_mae=1000.0)
    assert not should_roll_back(champion_mae=1040.0, previous_mae=1000.0)
