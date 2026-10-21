# Single-card throughput ceiling for StartLux Instant readout on Ascend.
#
# Instant inference = one big bidirectional prefill + a 1-token readout, so the
# ceiling is dominated by prefill throughput. This benchmark sweeps batch size
# and prompt length, then reports request rate and prompt tokens/s.
#
# Env knobs:
#   STARTLUX_BENCH_LENS   comma list of prompt token lengths   (default "512,1024,2048")
#   STARTLUX_BENCH_BATCH  comma list of batch sizes            (default "8,32,64")
#   STARTLUX_BENCH_WARM   warmup batches per cell              (default 1)
#   STARTLUX_BENCH_ITER   measured batches per cell            (default 2)
import os
import sys
import time

MODEL = os.environ.get(
    "STARTLUX_MODEL",
    "/export/lishuo121_test/startlux/models/StartLux-Decision-v1.1-0.8B-Instant",
)

os.environ.setdefault("VLLM_ASCEND_STARTLUX_PREFIX_BIDIR", "1")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

IMS = "<" + "|im_start|>"
IME = "<" + "|im_end|>"
THINK = "<think>"
QUESTION = (
    "Read the question and pick the correct option letter.\n\n"
    "Question: What is 2 + 2?\n"
    "A. 3\n"
    "B. 4\n"
    "C. 5\n"
    "D. 6\n"
)


def make_prompt(tok, n_tokens: int):
    """Prompt of exactly n_tokens ending with the Instant one-pass suffix."""
    head = f"{IMS}user\n{QUESTION}"
    tail = f"{IME}\n{IMS}assistant\n{THINK}\n\n</think>\n\n"
    h = tok.encode(head, add_special_tokens=False)
    t = tok.encode(tail, add_special_tokens=False)
    unit = tok.encode(" Please reason carefully. ", add_special_tokens=False)
    need = n_tokens - len(h) - len(t)
    filler = (unit * (need // len(unit) + 2))[: max(0, need)]
    ids = h + filler + t
    return ids[:n_tokens]


def main() -> int:
    lens = [int(x) for x in os.environ.get("STARTLUX_BENCH_LENS", "512,1024,2048").split(",")]
    batches = [int(x) for x in os.environ.get("STARTLUX_BENCH_BATCH", "8,32,64").split(",")]
    warm = int(os.environ.get("STARTLUX_BENCH_WARM", "1"))
    iters = int(os.environ.get("STARTLUX_BENCH_ITER", "2"))

    max_len = max(lens) + 64
    max_batched = max(65536, max_len)
    tok = AutoTokenizer.from_pretrained(MODEL)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=max_len,
        max_num_batched_tokens=max_batched,
        max_num_seqs=max(batches),
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=os.environ.get("STARTLUX_GRAPH", "") == "",
        compilation_config=(
            {"cudagraph_mode": os.environ["STARTLUX_GRAPH"].upper()} if os.environ.get("STARTLUX_GRAPH") else None
        ),
        gpu_memory_utilization=float(os.environ.get("STARTLUX_GPU_MEM", "0.9")),
        trust_remote_code=True,
        seed=0,
        disable_log_stats=True,
    )

    sts = llm.collective_rpc("startlux_prefix_status")
    s0 = sts[0]
    print(f"[bench] mode={s0['mode']} runner={s0['runner']}", flush=True)

    allowed = None
    if os.environ.get("STARTLUX_LETTERS") == "1":
        import json

        with open(os.path.join(MODEL, "decision_config.json")) as f:
            allowed = list(json.load(f)["letter_token_ids"])[:26]
        print(f"[bench] allowed_token_ids: {len(allowed)} letter ids", flush=True)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=1,
        detokenize=False,
        allowed_token_ids=allowed,
    )
    print(
        f"{'plen':>6} {'batch':>6} {'wall_s':>8} {'req/s':>8} {'ptok/s':>9} {'direct':>7} {'fixup':>6} {'lmhead':>6}",
        flush=True,
    )

    for plen in lens:
        ids = make_prompt(tok, plen)
        assert len(ids) == plen
        for bs in batches:
            prompts = [{"prompt_token_ids": ids} for _ in range(bs)]
            for _ in range(warm):
                llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
            t0 = time.perf_counter()
            for _ in range(iters):
                llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
            dt = (time.perf_counter() - t0) / iters
            s0 = llm.collective_rpc("startlux_prefix_status")[0]
            print(
                f"{plen:>6} {bs:>6} {dt:>8.2f} {bs / dt:>8.1f} {bs * plen / dt:>9.0f} "
                f"{s0['direct_calls']:>7} {s0['fixup_calls']:>6} {s0.get('lmhead_calls', 0):>6}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
