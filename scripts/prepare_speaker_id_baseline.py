from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber.baseline_prep import prepare_baseline  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare versioned speaker-ID baseline artifacts and canonical eval outputs."
    )
    parser.add_argument("recipe", type=Path, help="Path to the baseline prep YAML/JSON recipe.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root override. Defaults to the recipe's output_root.",
    )
    parser.add_argument(
        "--allow-legacy-reuse",
        action="store_true",
        help="Allow reuse of artifacts that predate manifests when the on-disk outputs exist.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume from reusable stage manifests when possible (default behavior unless disabled in the recipe).",
    )
    parser.add_argument(
        "--from-stage",
        choices=["bank", "mixed_base", "variants", "hard_negatives", "train", "eval"],
        default=None,
        help="Start execution from the named stage, reusing earlier stage manifests.",
    )
    parser.add_argument(
        "--to-stage",
        choices=["bank", "mixed_base", "variants", "hard_negatives", "train", "eval"],
        default=None,
        help="Stop execution after the named stage.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary = prepare_baseline(
        recipe_path=args.recipe,
        output_root=args.output_root,
        allow_legacy_reuse=bool(args.allow_legacy_reuse),
        resume=args.resume,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
