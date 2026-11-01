import pytest

from morpheus_policy.navigation import NavigationConfig
from morpheus_policy.optimization import MAX_CANDIDATES, MAX_SEEDS_PER_SPLIT, optimize


def _report(completed, duration=4.0, error=0.1, contacts=0, tilt=0.1):
    return {
        "execution": "completed" if completed else "failed",
        "metrics": {
            "duration_s": duration,
            "goal_error_m": error,
            "forbidden_contacts": contacts,
            "max_tilt_rad": tilt,
        },
    }


def test_train_only_selection_then_single_holdout_comparison():
    baseline = NavigationConfig(speed_m_s=0.10)
    winner = NavigationConfig(speed_m_s=0.20)
    rejected = NavigationConfig(speed_m_s=0.30)
    calls = []

    def evaluate(config, seed):
        calls.append((config.speed_m_s, seed))
        if config.speed_m_s == rejected.speed_m_s:
            return _report(False)
        return _report(True, duration=10.0 if config.speed_m_s == baseline.speed_m_s else 5.0)

    result = optimize(evaluate, baseline, [winner, rejected], [1, 2], [11, 12])

    assert result["selected_candidate_index"] == 0
    assert result["accepted"] is True
    assert result["chosen_config"] == result["selected_candidate_config"]
    # The rejected candidate is never evaluated on holdout; selection cannot leak.
    assert (rejected.speed_m_s, 11) not in calls
    assert (rejected.speed_m_s, 12) not in calls
    assert calls.count((winner.speed_m_s, 11)) == 1
    assert calls.count((baseline.speed_m_s, 11)) == 1


def test_holdout_regression_retains_baseline_and_evaluator_failure_is_audited():
    baseline = NavigationConfig(speed_m_s=0.10)
    regressor = NavigationConfig(speed_m_s=0.20)
    broken = NavigationConfig(speed_m_s=0.30)

    def evaluate(config, _seed):
        if config.speed_m_s == broken.speed_m_s:
            raise RuntimeError("simulator diverged")
        report = _report(config.speed_m_s == baseline.speed_m_s)
        if config.speed_m_s != baseline.speed_m_s:
            report["failures"] = ["forbidden contact"]
        return report

    result = optimize(evaluate, baseline, [regressor, broken], [1], [2])

    assert result["accepted"] is False
    assert result["chosen_config"] == {**result["chosen_config"], "speed_m_s": baseline.speed_m_s}
    failure = result["candidates"][1]["train"]["runs"][0]["failure"]
    assert failure == {"type": "RuntimeError", "message": "simulator diverged"}
    assert result["candidates"][0]["train"]["runs"][0]["failures"] == ["forbidden contact"]
    assert result["holdout"]["selected_candidate"]["completed"] == 0


def test_rejects_split_leakage_and_bound_violations():
    baseline = NavigationConfig()
    evaluate = lambda _config, _seed: _report(True)
    with pytest.raises(ValueError, match="disjoint"):
        optimize(evaluate, baseline, [baseline], [1], [1])
    with pytest.raises(ValueError, match="12"):
        optimize(evaluate, baseline, [baseline] * (MAX_CANDIDATES + 1), [1], [2])
    with pytest.raises(ValueError, match="at most"):
        optimize(evaluate, baseline, [baseline], list(range(MAX_SEEDS_PER_SPLIT + 1)), [99])
