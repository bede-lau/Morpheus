from pathlib import Path

from morpheus_policy.evaluation import Task
from morpheus_policy.qualification import summarize_qualification
from morpheus_policy.scene import Scene


_SCENE_HASH = "a" * 64
_MODEL_HASH = "b" * 64
_POLICY_HASH = "c" * 64


def _scene(*, scale=True, reviewed=True, synthetic=False, caveats=()):
    return Scene(
        "qualification fixture", (),
        {"scaleConfirmed": scale, "reviewed": reviewed, "synthetic": synthetic},
        Path("."), Path("scene.xml"), {"scene.json": _SCENE_HASH}, tuple(caveats),
    )


def _report(**changes):
    report = {
        "execution": "completed", "readiness": "PREVIEW_ONLY", "failures": [],
        "metrics": {"duration_s": 1.0, "goal_error_m": 0.1, "forbidden_contacts": 0, "max_tilt_rad": 0.1},
        "hashes": {"scene_json_sha256": _SCENE_HASH, "mjcf_sha256": _MODEL_HASH, "policy_sha256": _POLICY_HASH},
        "controller_config": {"speed_m_s": 0.3},
    }
    report.update(changes)
    return report


def _valid_reports():
    return [_report() for _ in range(20)]


def test_twenty_clean_consistent_reports_qualify_when_source_is_credible():
    result = summarize_qualification(_scene(caveats=("informative only",)), Task((0, 0), (1, 0)), _valid_reports(), seed=1)

    assert result["execution"] == "completed"
    assert result["readiness"] == "SIM_READY"
    assert result["coverage"]["successes"] == 20
    assert result["source_disqualifiers"] == []


def test_completed_report_with_forbidden_contact_is_blocked():
    reports = _valid_reports()
    reports[4] = _report(metrics={**reports[4]["metrics"], "forbidden_contacts": 1})
    result = summarize_qualification(_scene(), Task((0, 0), (1, 0)), reports, seed=1)

    assert result["execution"] == "failed"
    assert result["readiness"] == "BLOCKED"
    assert any("episode[4]: forbidden_contacts" in reason for reason in result["disqualifiers"])


def test_mismatched_provenance_or_controller_config_is_blocked():
    reports = _valid_reports()
    reports[2] = _report(hashes={**reports[2]["hashes"], "policy_sha256": "d" * 64})
    reports[3] = _report(controller_config={"speed_m_s": 0.2})
    result = summarize_qualification(_scene(), Task((0, 0), (1, 0)), reports, seed=1)

    assert result["execution"] == "failed"
    assert any("hashes differ" in reason for reason in result["disqualifiers"])
    assert any("controller_config differs" in reason for reason in result["disqualifiers"])


def test_missing_or_nonfinite_required_metric_is_blocked():
    reports = _valid_reports()
    reports[0] = _report(metrics={"goal_error_m": 0.1, "forbidden_contacts": 0, "max_tilt_rad": 0.1})
    reports[1] = _report(metrics={"duration_s": float("nan"), "goal_error_m": 0.1, "forbidden_contacts": 0, "max_tilt_rad": 0.1})
    result = summarize_qualification(_scene(), Task((0, 0), (1, 0)), reports, seed=1)

    assert result["execution"] == "failed"
    assert any("episode[0]: metric duration_s" in reason for reason in result["disqualifiers"])
    assert any("episode[1]: metric duration_s" in reason for reason in result["disqualifiers"])


def test_source_flags_and_synthetic_task_limit_readiness_without_erasing_success():
    result = summarize_qualification(_scene(scale=False, reviewed=False, synthetic=True), Task((0, 0), (1, 0)), _valid_reports(), seed=1)
    assert result["execution"] == "completed"
    assert result["readiness"] == "PREVIEW_ONLY"
    assert len(result["source_disqualifiers"]) == 3

    synthetic_task = Task((0, 0), (1, 0), synthetic=True)
    task_result = summarize_qualification(_scene(), synthetic_task, _valid_reports(), seed=1)
    assert task_result["execution"] == "completed"
    assert task_result["readiness"] == "PREVIEW_ONLY"
    assert task_result["source_disqualifiers"] == ["Task is synthetic"]
