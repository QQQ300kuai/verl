"""
Convert the LlamaFactory sharegpt-format e-commerce tool-calling dataset
(ecommerce_cs_toolcall.json) into a verl GRPO parquet dataset with
rule-verifiable ground truth.

Why only ecommerce_cs_toolcall.json:
- ecommerce_cs_policy.json / ecommerce_cs_sharegpt.json are open-ended
  policy Q&A / chit-chat with no ground truth that can be checked by a rule,
  and we have no reward model / judge model to score them. Feeding them into
  GRPO without a real reward signal would just add noise.
- ecommerce_cs_toolcall.json has a clean, checkable target: for each turn we
  know whether the model *should* call a tool, and if so, which tool with
  which arguments (extracted from the original human -> function_call ->
  observation -> gpt trace). That is exactly the RLVR shape GRPO is good at,
  and tool-call accuracy is usually the weakest link after a LoRA SFT.

The tools block is rendered with the *actual* Qwen tokenizer's
apply_chat_template(..., tools=...) so it is byte-identical to what verl
will render again (without `tools=`) at rollout time -- we bake the tools
text into a plain "system" message instead of relying on verl's per-row
tool-schema wiring (which is designed for live multi-turn tool execution,
not needed here since we only score a single generated turn).

Usage (run on the server where the base model + LlamaFactory data live):

    python3 prepare_data.py \
        --model_path /root/autodl-tmp/models/Qwen/Qwen3-4B-Instruct-2507 \
        --toolcall_json /root/autodl-tmp/code/LlamaFactory/data/ecommerce_cs_toolcall.json \
        --save_dir ./data/ \
        --val_ratio 0.1 --seed 42
"""

import argparse
import json
import os
import re

import pandas as pd


def extract_system_block(tokenizer, tools: list[dict]) -> str:
    """Render a throwaway single-turn conversation with `tools=` and pull out
    just the system message text, so we can re-embed it as a plain message."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "__PLACEHOLDER__"}],
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    )
    match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", rendered, re.DOTALL)
    if not match:
        raise ValueError("Could not locate a system block in the rendered chat template output; check the template.")
    return match.group(1)


def build_examples(raw_data: list[dict], tokenizer):
    examples = []
    skipped = 0
    for idx, item in enumerate(raw_data):
        conv = item["conversations"]
        tools = json.loads(item["tools"]) if item.get("tools") else []

        if len(conv) == 2 and conv[0]["from"] == "human" and conv[1]["from"] == "gpt":
            expect_call = False
            question = conv[0]["value"]
            ground_truth = {"expect_call": False}
        elif (
            len(conv) == 4
            and conv[0]["from"] == "human"
            and conv[1]["from"] == "function_call"
            and conv[2]["from"] == "observation"
            and conv[3]["from"] == "gpt"
        ):
            expect_call = True
            question = conv[0]["value"]
            try:
                call = json.loads(conv[1]["value"])
            except json.JSONDecodeError:
                skipped += 1
                continue
            ground_truth = {
                "expect_call": True,
                "name": call.get("name"),
                "arguments": call.get("arguments", {}),
            }
        else:
            # unexpected conversation shape (multi-round tool use, etc.) -- skip,
            # this dataset only has the two shapes above today.
            skipped += 1
            continue

        system_content = extract_system_block(tokenizer, tools)

        examples.append(
            {
                "data_source": "ecommerce_cs_toolcall",
                "prompt": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": question},
                ],
                "ability": "tool_calling",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
                "extra_info": {
                    "index": idx,
                    "expect_call": expect_call,
                    "question": question,
                },
            }
        )

    if skipped:
        print(f"skipped {skipped} examples with an unrecognized conversation shape")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Base model path (tokenizer must support tools=).")
    parser.add_argument("--toolcall_json", required=True, help="Path to ecommerce_cs_toolcall.json")
    parser.add_argument("--save_dir", required=True, help="Output dir for train.parquet / test.parquet")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.toolcall_json, encoding="utf-8") as f:
        raw_data = json.load(f)

    examples = build_examples(raw_data, tokenizer)

    df = pd.DataFrame(examples)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_val = max(1, int(len(df) * args.val_ratio))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)

    pos = sum(1 for e in examples if e["extra_info"]["expect_call"])
    neg = len(examples) - pos
    print(f"total examples: {len(examples)} (expect_call=True: {pos}, expect_call=False: {neg})")
    print(f"train: {len(train_df)}, val: {len(val_df)}")

    os.makedirs(args.save_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    val_df.to_parquet(os.path.join(args.save_dir, "test.parquet"))
    print(f"wrote parquet files to {args.save_dir}")


if __name__ == "__main__":
    main()
