"""Short end-to-end SAC diagnostic; this is not a trained Stage-1 result."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_zl.training_config import load_stage1_config


def main() -> int:
    config = load_stage1_config(ROOT / "configs" / "stage1_teacher.yaml")
    replay = replace(
        config.replay,
        capacity=2048,
        batch_size=32,
        learning_starts=32,
        update_after=32,
    )
    training = replace(
        config.training,
        evaluation_interval_steps=128,
        evaluation_episodes=2,
        final_evaluation_episodes=2,
        checkpoint_interval_steps=128,
        log_interval_steps=64,
    )
    curriculum_stage = replace(
        config.curriculum[0],
        minimum_training_steps=0,
        maximum_training_steps=128,
        promotion_success_rate=1.0,
    )
    config = replace(
        config,
        output_dir=ROOT / "artifacts" / "stage1_smoke",
        replay=replay,
        training=training,
        curriculum=(curriculum_stage,),
    )
    try:
        from rl_zl.training import Stage1Trainer
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    trainer = Stage1Trainer(config)
    result = trainer.run(max_steps_override=128, final_evaluation_episodes_override=2)
    result["smoke_passed"] = bool(
        result["total_steps"] == 128
        and result["updates"] > 0
        and Path(result["final_checkpoint"]).exists()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
