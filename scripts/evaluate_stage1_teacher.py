"""Evaluate a Stage-1 teacher checkpoint on fixed, failure-retaining episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_zl.evaluation import evaluate_policy
from rl_zl.training_config import apply_curriculum_stage, load_stage1_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_teacher.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--curriculum-index", type=int, default=-1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seed-split",
        choices=("validation", "confirmation", "final_test"),
        default="final_test",
        help="Use a documented, disjoint evaluation seed block.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Explicitly override the selected seed block (recorded in output).",
    )
    parser.add_argument("--output", default="artifacts/stage1_teacher/evaluation_manual.json")
    args = parser.parse_args()

    config = load_stage1_config(ROOT / args.config)
    stage_index = (
        args.curriculum_index
        if args.curriculum_index >= 0
        else len(config.curriculum) + args.curriculum_index
    )
    if not 0 <= stage_index < len(config.curriculum):
        parser.error("--curriculum-index is outside the configured curriculum")
    stage = config.curriculum[stage_index]
    environment_config = apply_curriculum_stage(config.load_base_environment(), stage)
    try:
        from rl_zl.sac import SACAgent
    except ImportError as exc:
        parser.error(str(exc))
    agent = SACAgent(53, 3, config.sac, device=config.device)
    agent.load(ROOT / args.checkpoint, load_optimizers=False)
    configured_seeds = {
        "validation": config.training.validation_seed,
        "confirmation": config.training.confirmation_seed,
        "final_test": config.training.final_test_seed,
    }
    base_seed = int(
        args.base_seed
        if args.base_seed is not None
        else configured_seeds[args.seed_split] + stage_index * 100_000
    )
    summary = evaluate_policy(
        agent,
        environment_config,
        episodes=args.episodes,
        base_seed=base_seed,
        deterministic=True,
    )
    result = {
        "stage": stage.name,
        "checkpoint": args.checkpoint,
        "seed_split": args.seed_split,
        "base_seed": base_seed,
        **summary.to_dict(include_records=True),
        "stage1_gate_passed": summary.stage1_gate_passes(),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
