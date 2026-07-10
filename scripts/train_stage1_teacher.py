"""Train the V4 Stage-1 SAC teacher with curriculum gates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_zl.training_config import load_stage1_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_teacher.yaml")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Diagnostic override; not a V4 result")
    parser.add_argument("--final-eval-episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_stage1_config(ROOT / args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.output_dir is not None:
        config = replace(config, output_dir=(ROOT / args.output_dir).resolve())

    try:
        from rl_zl.training import Stage1Trainer
    except ImportError as exc:
        parser.error(str(exc))
    trainer = Stage1Trainer(config)
    if args.resume is not None:
        trainer.resume(ROOT / args.resume)
    result = trainer.run(
        max_steps_override=args.max_steps,
        final_evaluation_episodes_override=args.final_eval_episodes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
