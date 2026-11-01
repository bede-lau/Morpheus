"""Bounded selection of a frozen actor's navigation-controller configuration.

This module deliberately searches controller parameters only.  It never trains,
replaces, or observes the learned locomotion actor.
"""

from dataclasses import asdict
import math
from typing import Any, Callable, Mapping

from .navigation import NavigationConfig


MAX_CANDIDATES = 12
MAX_SEEDS_PER_SPLIT = 20
FAILURE_SCORE_PENALTY = 1_000_000_000.0
_METRICS = ("duration_s", "goal_error_m", "forbidden_contacts", "max_tilt_rad")


def _validate_split(train_seeds: list[int], holdout_seeds: list[int]) -> None:
    if not train_seeds or not holdout_seeds:
        raise ValueError("Training and holdout seed splits must both be non-empty")
    if len(train_seeds) > MAX_SEEDS_PER_SPLIT or len(holdout_seeds) > MAX_SEEDS_PER_SPLIT:
        raise ValueError(f"Each split supports at most {MAX_SEEDS_PER_SPLIT} seeds")
    if len(set(train_seeds)) != len(train_seeds) or len(set(holdout_seeds)) != len(holdout_seeds):
        raise ValueError("Seed splits must not contain duplicate seeds")
    if set(train_seeds) & set(holdout_seeds):
        raise ValueError("Training and holdout seeds must be disjoint")


def _completed_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise ValueError("Evaluation must return a mapping")
    execution = report.get("execution")
    if execution not in {"completed", "failed"}:
        raise ValueError("Evaluation report execution must be completed or failed")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Evaluation report requires metrics")
    normalized = {}
    for name in _METRICS:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Evaluation metric {name} must be a finite non-negative number")
        normalized[name] = float(value)
    preserved = {"execution": execution, "metrics": normalized}
    # Evaluation owns domain failure semantics (contacts, divergence, missing
    # assets). Keep them inspectable rather than reducing failed episodes to a
    # scalar score.
    for key in ("failure", "failures", "caveats"):
        if key in report:
            preserved[key] = report[key]
    return preserved


def _run(evaluate_config: Callable[[NavigationConfig, int], Mapping[str, Any]], config: NavigationConfig, seed: int) -> dict[str, Any]:
    """Translate evaluator exceptions and malformed reports into retained failures."""
    try:
        report = _completed_report(evaluate_config(config, seed))
    except Exception as exc:  # Preserve a per-seed failure without abandoning the bounded search.
        return {
            "seed": seed,
            "execution": "failed",
            "metrics": None,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {"seed": seed, **report}


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run["execution"] == "completed"]
    values = {name: sum(run["metrics"][name] for run in completed) for name in _METRICS}
    count = len(runs)
    mean = {name: (values[name] / len(completed) if completed else None) for name in _METRICS}
    # Failures rank below completed runs before this tie-break is consulted.  A
    # finite penalty keeps the returned audit record valid JSON, unlike Infinity.
    scoring_total = {name: values[name] + (count - len(completed)) * FAILURE_SCORE_PENALTY for name in _METRICS}
    score = (
        len(completed),
        -scoring_total["forbidden_contacts"],
        -scoring_total["goal_error_m"],
        -scoring_total["max_tilt_rad"],
        -scoring_total["duration_s"],
    )
    return {
        "episodes": count,
        "completed": len(completed),
        "failed": count - len(completed),
        "total": values,
        "mean": mean,
        "failure_score_penalty": FAILURE_SCORE_PENALTY,
        "success_first_score": list(score),
        "runs": runs,
    }


def optimize(
    evaluate_config: Callable[[NavigationConfig, int], Mapping[str, Any]],
    baseline: NavigationConfig,
    candidates: list[NavigationConfig],
    train_seeds: list[int],
    holdout_seeds: list[int],
) -> dict[str, Any]:
    """Select on train only, then compare one selected config with baseline once.

    The return value is a complete audit record.  A candidate is accepted only
    when its *held-out* success-first score is strictly better than the baseline.
    Equal outcomes retain the baseline rather than claiming an improvement.
    """
    if not callable(evaluate_config):
        raise TypeError("evaluate_config must be callable")
    if not candidates or len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"Supply between one and {MAX_CANDIDATES} candidate configurations")
    _validate_split(train_seeds, holdout_seeds)

    candidate_records = []
    for index, config in enumerate(candidates):
        train_runs = [_run(evaluate_config, config, seed) for seed in train_seeds]
        candidate_records.append({
            "index": index,
            "config": asdict(config),
            "train": _summarize(train_runs),
        })

    # Lower candidate index is deterministic tie-breaking; held-out results are
    # intentionally absent from this selector.
    selected = max(
        candidate_records,
        key=lambda record: (tuple(record["train"]["success_first_score"]), -record["index"]),
    )
    selected_config = candidates[selected["index"]]
    candidate_holdout = _summarize([_run(evaluate_config, selected_config, seed) for seed in holdout_seeds])
    baseline_holdout = _summarize([_run(evaluate_config, baseline, seed) for seed in holdout_seeds])
    accepted = tuple(candidate_holdout["success_first_score"]) > tuple(baseline_holdout["success_first_score"])

    return {
        "optimization": "frozen_actor_navigation_controller_config_search",
        "actor": "frozen",
        "limits": {"max_candidates": MAX_CANDIDATES, "max_seeds_per_split": MAX_SEEDS_PER_SPLIT},
        "train_seeds": list(train_seeds),
        "holdout_seeds": list(holdout_seeds),
        "candidates": candidate_records,
        "selected_candidate_index": selected["index"],
        "selected_candidate_config": asdict(selected_config),
        "holdout": {"selected_candidate": candidate_holdout, "baseline": baseline_holdout},
        "accepted": accepted,
        "chosen_config": asdict(selected_config if accepted else baseline),
        "acceptance_reason": (
            "Selected controller strictly improved the held-out success-first score"
            if accepted
            else "Baseline retained: selected controller did not strictly improve held-out score"
        ),
    }
