"""Case 加载与校验脚本。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = {"case_id", "language", "length", "messages", "boundaries"}
VALID_LANGUAGES = {"zh", "en"}
VALID_LENGTHS = {"short", "long"}
VALID_ROLES = {"user", "assistant", "tool", "system"}


def load_case(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cases(cases_dir: str | Path = "eval/cases") -> list[dict]:
    return [load_case(p) for p in sorted(Path(cases_dir).glob("*.json"))]


def count_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def validate_case(case: dict) -> list[str]:
    """返回错误列表，空列表表示合法。"""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in case:
            errors.append(f"missing field: {field}")
    if errors:
        return errors

    if case["language"] not in VALID_LANGUAGES:
        errors.append(f"invalid language: {case['language']}")
    if case["length"] not in VALID_LENGTHS:
        errors.append(f"invalid length: {case['length']}")

    messages = case["messages"]
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
        return errors

    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            errors.append(f"message {i} missing role/content")
        elif m["role"] not in VALID_ROLES:
            errors.append(f"message {i} invalid role: {m['role']}")

    turns = count_turns(messages)
    if turns < 2:
        errors.append("need at least 2 user turns")

    boundaries = case["boundaries"]
    if not isinstance(boundaries, list):
        errors.append("boundaries must be a list")
    else:
        for b in boundaries:
            if not isinstance(b, int) or not (0 < b < turns):
                errors.append(f"invalid boundary: {b!r} (turns={turns})")
        if sorted(boundaries) != boundaries:
            errors.append("boundaries must be strictly ascending")

    return errors


def check_dimension_coverage(cases: list[dict]) -> dict[tuple[str, str], int]:
    coverage: dict[tuple[str, str], int] = {}
    for c in cases:
        key = (c["language"], c["length"])
        coverage[key] = coverage.get(key, 0) + 1
    return coverage


def main(cases_dir: str = "eval/cases") -> int:
    cases = load_cases(cases_dir)
    ok = True

    for c in cases:
        errors = validate_case(c)
        cid = c.get("case_id", "?")
        if errors:
            ok = False
            print(f"[FAIL] {cid}: {errors}")
        else:
            print(f"[OK] {cid} ({c['language']}-{c['length']}, {count_turns(c['messages'])} turns)")

    coverage = check_dimension_coverage(cases)
    print("coverage:", dict(sorted(coverage.items())))

    for lang in sorted(VALID_LANGUAGES):
        for length in sorted(VALID_LENGTHS):
            n = coverage.get((lang, length), 0)
            if n < 2:
                ok = False
                print(f"[MISSING] {lang}-{length} has {n} case(s), need >= 2")

    print("ALL OK" if ok else "HAS ISSUES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
