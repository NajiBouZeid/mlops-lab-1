"""
Quality gate: promote a registered model version to `champion` only if it is
clearly better than the current champion.

Usage:
    uv run python src/food11/promote.py --version 3
"""

import argparse
import math
import os
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "food11"
METRIC = "test_accuracy"
DEFAULT_TEST_DIR = Path("data/food11_processed_mini/evaluation")  # the split train.py reports test_accuracy on


def count_images(test_dir: Path) -> int:
    return sum(1 for p in test_dir.rglob("*") if p.is_file())


def standard_error(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def decide(acc_candidate: float, acc_champion: float, n: int) -> tuple[bool, float]:
    """Returns (promote?, threshold the candidate had to reach)."""
    threshold = acc_champion + 2 * standard_error(acc_champion, n)
    return acc_candidate >= threshold, threshold


def run_accuracy(client: MlflowClient, run_id: str) -> float:
    metrics = client.get_run(run_id).data.metrics
    if METRIC not in metrics:
        raise SystemExit(f"Run {run_id} has no '{METRIC}' metric")
    return metrics[METRIC]


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model version if it passes the quality gate")
    parser.add_argument("--version", required=True, help="candidate model version number")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    n = count_images(args.test_dir)
    if n == 0:
        raise SystemExit(f"No test images found in {args.test_dir}")

    candidate = client.get_model_version(MODEL_NAME, args.version)
    acc_candidate = run_accuracy(client, candidate.run_id)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:  # no version holds the alias yet
        champion = None

    if champion is None:
        promote, threshold, acc_champion = True, 0.0, float("nan")
        print("No current champion: promoting the candidate.")
    elif champion.version == candidate.version:
        raise SystemExit(f"Version {args.version} is already the champion")
    else:
        acc_champion = run_accuracy(client, champion.run_id)
        promote, threshold = decide(acc_candidate, acc_champion, n)

    se = standard_error(acc_champion, n) if champion is not None else float("nan")
    print(f"n (test images)       : {n}")
    print(f"champion              : v{champion.version if champion else '-'}  {METRIC}={acc_champion:.4f}  se={se:.4f}")
    print(f"candidate             : v{candidate.version}  {METRIC}={acc_candidate:.4f}")
    print(f"threshold (champ+2se) : {threshold:.4f}")

    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
        decision = "promoted to champion"
    else:
        client.set_registered_model_alias(MODEL_NAME, "challenger", candidate.version)
        decision = "kept as challenger"
    print(f"decision              : {decision}")

    for key, value in {
        "gate_decision": decision,
        "gate_candidate_accuracy": f"{acc_candidate:.4f}",
        "gate_champion_accuracy": f"{acc_champion:.4f}",
        "gate_threshold": f"{threshold:.4f}",
        "gate_n_test": str(n),
    }.items():
        client.set_model_version_tag(MODEL_NAME, candidate.version, key, value)


if __name__ == "__main__":
    main()