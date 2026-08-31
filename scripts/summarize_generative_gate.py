#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--id-run", type=Path, required=True)
    parser.add_argument("--content-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    alignment = json.loads((args.alignment / "metrics.json").read_text())
    id_result = json.loads((args.id_run / "metrics.json").read_text())
    content = json.loads((args.content_run / "metrics.json").read_text())
    alignment_pass = bool(alignment["gate_passed"])
    content_gain = (
        content["valid"]["recall@40"] - id_result["valid"]["recall@40"]
    )
    candidate_safe = (
        id_result["candidate_authors"] == content["candidate_authors"]
        and not content["audit"]["candidate_universe_reduced"]
    )
    result = {
        "alignment_gate_passed": alignment_pass,
        "candidate_universe_safe": candidate_safe,
        "id_valid_recall@40": id_result["valid"]["recall@40"],
        "content_valid_recall@40": content["valid"]["recall@40"],
        "content_valid_recall@40_gain": content_gain,
        "id_test": id_result["test"],
        "content_test": content["test"],
        "gate_passed": bool(alignment_pass and candidate_safe and content_gain > 0),
        "next_step": "replicate and compare tuned SASRec" if (
            alignment_pass and candidate_safe and content_gain > 0
        ) else "stop before multi-seed GRPO expansion",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
