"""
Utilities for parsing JSON from LLM responses.

Open-source models often wrap JSON in markdown fences or add leading text.
These helpers strip that noise before parsing.
"""

import json
import re


def parse_json_response(text: str) -> dict:
    """
    Parse a JSON object from an LLM response string.

    Handles:
    - Plain JSON strings
    - JSON wrapped in ```json ... ``` or ``` ... ``` fences
    - Leading / trailing prose around the JSON object

    Raises:
        ValueError: if no valid JSON object can be extracted.
    """
    if not text or not text.strip():
        raise ValueError("LLM returned an empty response")

    cleaned = text.strip()

    # Strip markdown fences: ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    match = fence_pattern.search(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # If still not starting with '{', find the first '{' (handles leading prose)
    brace_start = cleaned.find("{")
    if brace_start == -1:
        raise ValueError(
            f"No JSON object found in LLM response. Raw text:\n{text[:500]}"
        )
    cleaned = cleaned[brace_start:]

    # Find the matching closing brace
    brace_end = _find_matching_brace(cleaned)
    if brace_end == -1:
        raise ValueError(
            f"Could not find closing brace in LLM response. Raw text:\n{text[:500]}"
        )
    cleaned = cleaned[: brace_end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Second attempt: fix unescaped backslashes inside string values
    # (common when LLMs embed Python code with \n, \t, etc. in a JSON string)
    try:
        fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
        return json.loads(fixed)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON parse error: {exc}. Extracted text:\n{cleaned[:500]}"
        ) from exc


def _find_matching_brace(text: str) -> int:
    """Return index of the closing '}' that matches the first '{'."""
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1
