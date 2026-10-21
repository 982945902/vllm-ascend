# Smoke test: StartLux Instant bidirectional readout on Ascend.
#
# Runs one Instant-style prompt (ending with think + blank think close, so the
# whole prompt is read bidirectionally, p == L) through the vendored
# vllm-ascend and prints the option-letter readout plus the patch's status
# counters. With STARTLUX_PREFIX_BIDIR_CHECK=1 every fixed-up layer call is
# also verified against an fp32 explicit-mask reference.
#
# Required env:
#   VLLM_ASCEND_STARTLUX_PREFIX_BIDIR=1  (or STARTLUX_PREFIX_BIDIR=1)
#   ASCEND_RT_VISIBLE_DEVICES=<free chip>
import os
import sys

MODEL = os.environ.get(
    "STARTLUX_MODEL",
    "/export/lishuo121_test/startlux/models/StartLux-Decision-v1.1-0.8B-Instant",
)
MAX_LEN = int(os.environ.get("STARTLUX_MAX_LEN", "4096"))

os.environ.setdefault("VLLM_ASCEND_STARTLUX_PREFIX_BIDIR", "1")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

IMS = "<" + "|im_start|>"  # literal special token kept out of this file
IME = "<" + "|im_end|>"
THINK = "<think>"
PROMPT = (
    f"{IMS}user\n"
    "Read the question and pick the correct option letter.\n\n"
    "Question: What is 2 + 2?\n"
    "A. 3\n"
    "B. 4\n"
    "C. 5\n"
    "D. 6\n"
    f"{IME}\n"
    f"{IMS}assistant\n"
    f"{THINK}\n\n</think>\n\n"
)


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.encode(PROMPT, add_special_tokens=False)
    print(f"[smoke] {len(ids)} prompt tokens", flush=True)

    llm_kwargs = {}
    if os.environ.get("STARTLUX_NO_MM") == "1":
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if os.environ.get("STARTLUX_CPU_OFFLOAD"):
        llm_kwargs["cpu_offload_gb"] = float(os.environ["STARTLUX_CPU_OFFLOAD"])
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=MAX_LEN,
        max_num_batched_tokens=MAX_LEN,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("STARTLUX_GPU_MEM", "0.05")),
        trust_remote_code=True,
        seed=0,
        **llm_kwargs,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, detokenize=False)
    out = llm.generate(
        prompts=[{"prompt_token_ids": ids}],
        sampling_params=sp,
        use_tqdm=False,
    )[0]

    lp = out.outputs[0].logprobs[0]
    letters = {}
    for letter in "ABCDEFGH":
        tid = tok.convert_tokens_to_ids(letter)
        if tid in lp:
            letters[letter] = float(lp[tid].logprob)
    top = max(letters, key=letters.get) if letters else None
    print(f"[smoke] letter logprobs: {letters}")
    print(f"[smoke] readout answer: {top} (expected B)")

    sts = llm.collective_rpc("startlux_prefix_status")
    print(f"[smoke] plugin status: {sts}")
    s0 = sts[0]
    assert s0["mode"] == "bidirectional", s0
    # Pure Instant steps take the direct non-causal path (direct_calls); mixed
    # or think-style prompts fall back to causal + fix-up (fixup_calls).
    assert s0["direct_calls"] + s0["fixup_calls"] > 0, "bidirectional path never engaged"
    assert s0["bi_rows"] == s0["bi_requests"] * len(ids), (s0["bi_rows"], len(ids))
    print("[smoke] PASS: Instant bidirectional readout ran on Ascend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
