"""
Rule-based reward for the e-commerce customer-service tool-calling task.

No reward model / judge model involved: the ground truth (whether a tool
call was expected, and if so which tool + which arguments) was extracted
directly from the LlamaFactory SFT traces in prepare_data.py, so scoring is
just parsing the model's output and diffing it against that ground truth.

Score components (all in [0, 1]):
- expect_call == False (a plain policy question, no tool needed):
    1.0 if the response contains no <tool_call> block, else 0.0
    (penalizes hallucinated / unnecessary tool calls).
- expect_call == True:
    0.0  if no <tool_call> block is present (missed call), or the block is
         not valid JSON (0.1, small credit for at least attempting the format)
    0.5 * (tool name matches ground truth)
  + 0.5 * (fraction of ground-truth arguments reproduced with equal value)

Wire this up in the GRPO launch script with:
    custom_reward_function.path=.../reward_fn.py
    custom_reward_function.name=compute_score
"""

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _extract_tool_call(solution_str: str):
    match = _TOOL_CALL_RE.search(solution_str)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return "invalid_json"


def _normalize(value) -> str:
    return str(value).strip().lower()


def _arguments_match_fraction(predicted: dict, expected: dict) -> float:
    if not expected:
        return 1.0
    if not isinstance(predicted, dict):
        return 0.0
    matched = sum(1 for k, v in expected.items() if k in predicted and _normalize(predicted[k]) == _normalize(v))
    return matched / len(expected)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = json.loads(ground_truth)
    call = _extract_tool_call(solution_str)

    if not gt.get("expect_call"):
        return 1.0 if call is None else 0.0

    if call is None:
        return 0.0
    if call == "invalid_json":
        return 0.1

    name_score = 1.0 if call.get("name") == gt.get("name") else 0.0
    args_score = _arguments_match_fraction(call.get("arguments", {}), gt.get("arguments", {}))
    return 0.5 * name_score + 0.5 * args_score
