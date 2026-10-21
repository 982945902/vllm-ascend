# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend port of the StartLux-Decision prompt-bidirectional plugin.

Upstream reference: startlux_decide/vllm_plugin.py (Apache-2.0), which patches
vLLM's GPU FlashAttention path. This module applies the same contract to the
Ascend attention backend so that Instant model folders
(config.json: "jev_prompt_attention": "bidirectional") read their prompt
bidirectionally on NPU:

    query q may read key k   <=>   k <= q   or   (q < p and k < p)

in the FULL-ATTENTION layers only. Gated DeltaNet layers, their short
convolutions and the KV-cache write are untouched.

p is the prompt end of a request, derived from its prompt token ids:
    * prompt ends with "<think>\n\n</think>\n\n" (one-pass read-out): p = L
    * otherwise p = the position right after the "<think>\n" following the
      first "<|im_start|>assistant\n" header
    * SamplingParams(extra_args={"startlux_prompt_end": p}) overrides
    * no match: raise (STARTLUX_PREFIX_BIDIR_UNMATCHED=causal|full degrades)

How. Every full-attention layer first runs the stock causal FIA call of the
step. For each request prefilled in this step, rows [start, start + p) are
recomputed by one non-causal varlen npu_fusion_attention call over that
request's own prompt keys/values and written over the causal rows. Decode
steps are untouched.

Activation: STARTLUX_PREFIX_BIDIR=1 (the upstream plugin contract, so
startlux_decide tooling works unchanged) or VLLM_ASCEND_STARTLUX_PREFIX_BIDIR=1.
The feature only engages for folders whose config.json declares
"jev_prompt_attention": "bidirectional"; anything else runs stock.
"""

from __future__ import annotations

import functools
import importlib.abc
import inspect
import json
import logging
import os
import sys
import threading

logger = logging.getLogger("vllm_ascend.startlux_prefix_bidir")

ENV = "STARTLUX_PREFIX_BIDIR"
ENV_ASCEND = "VLLM_ASCEND_STARTLUX_PREFIX_BIDIR"
LEAN_ROWS = 16384  # prompt rows in a step above which the fix-up works in place
EXTRA_KEY = "startlux_prompt_end"
THINK = "<think>"
ONE_PASS_SUFFIX = "<think>\n\n</think>\n\n"
HEADER = "<|im_start|>assistant\n"
PROMPT_ATTENTION_KEY = "jev_prompt_attention"
ACTIVE_LINE = "prefix-bidirectional prompt attention ON (ascend)"

_lock = threading.Lock()
STATUS = {
    "installed": False,
    "hooked": [],
    "patched": [],
    "error": None,
    "pid": os.getpid(),
    "runners": 0,
    "bi_runners": 0,
    "steps_with_plan": 0,
    "bi_requests": 0,
    "bi_rows": 0,
    "max_p": 0,
    "explicit": 0,
    "one_pass": 0,
    "direct_calls": 0,
    "think": 0,
    "unmatched": 0,
    "fixup_calls": 0,
    "lmhead_calls": 0,
    "lmhead_cols": 0,
    "resumed_prefills": 0,
    "warmup_steps": 0,
    "check_calls": 0,
    "check_max_diff_prompt": 0.0,
    "check_max_diff_rest": 0.0,
    "check_min_bidir_effect": None,
    "check_max_stock_err": 0.0,
    "check_max_ratio": 0.0,
}


def env(suffix: str = "", default: str | None = None) -> str | None:
    # Debug/tuning knobs keep the upstream STARTLUX_PREFIX_BIDIR_* names so the
    # upstream tooling and docs apply verbatim.
    return os.environ.get(ENV + suffix, default)


def enabled() -> bool:
    if os.environ.get(ENV) == "1":
        return True
    try:
        from vllm_ascend import envs

        return bool(envs.VLLM_ASCEND_STARTLUX_PREFIX_BIDIR)
    except Exception:
        return False


def _say(msg: str) -> None:
    line = f"[startlux prefix-bidir ascend pid={os.getpid()}] {msg}"
    logger.info(line)
    print(line, file=sys.stderr, flush=True)


def read_prompt_attention(model_dir: str) -> str:
    """config.json["jev_prompt_attention"]; absent means causal."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return json.load(f).get(PROMPT_ATTENTION_KEY, "causal")
    except Exception:
        return "causal"


# ------------------------------------------------------------------- prompt end
class PromptEnd:
    """p of a prompt from its token ids. __call__(ids) -> (p or None, kind)."""

    def __init__(self, tokenizer):
        def enc(s: str) -> list:
            return list(tokenizer.encode(s, add_special_tokens=False))

        self.think = tokenizer.convert_tokens_to_ids(THINK)
        nl = enc("\n")
        if len(nl) != 1 or self.think is None or self.think < 0:
            raise ValueError("tokenizer: '<think>' or '\\n' is not a single token")
        self.nl = nl[0]
        self.suffix = enc(ONE_PASS_SUFFIX)
        self.header = enc(HEADER)
        if self.suffix[0] != self.think or self.suffix.count(self.think) != 1:
            raise ValueError(f"unexpected one-pass suffix tokens {self.suffix}")

    def __call__(self, ids) -> tuple:
        n, s, h = len(ids), len(self.suffix), len(self.header)
        if n >= s and list(ids[n - s :]) == self.suffix:
            return n, "one_pass"
        for j, t in enumerate(ids):
            if (
                t == self.think
                and j + 1 < n
                and ids[j + 1] == self.nl
                and j >= h
                and list(ids[j - h : j]) == self.header
            ):
                return j + 2, "think"
        return None, "unmatched"


# ------------------------------------------------------------------ runner side
class _RunnerState:
    def __init__(self, runner, kind: str):
        mc = runner.model_config
        self.model = str(mc.model)
        self.mode = read_prompt_attention(self.model) if os.path.isdir(self.model) else "causal"
        hf_mode = getattr(getattr(mc, "hf_config", None), PROMPT_ATTENTION_KEY, None)
        if hf_mode is not None and hf_mode != self.mode:
            raise RuntimeError(
                f"{ENV}: {PROMPT_ATTENTION_KEY} differs between {self.model}/config.json "
                f"({self.mode}) and the loaded hf_config ({hf_mode})"
            )
        self.bi = self.mode == "bidirectional"
        self.kind = kind
        self.tokenizer_path = str(mc.tokenizer or mc.model)
        self.device = runner.device
        self._pe = None
        self.logged = 0
        self.check = env("_CHECK") == "1"

    def prompt_end(self) -> PromptEnd:
        if self._pe is None:
            from transformers import AutoTokenizer

            self._pe = PromptEnd(AutoTokenizer.from_pretrained(self.tokenizer_path))
        return self._pe


def _check_runner_config(runner) -> None:
    """The engine settings the prefix mask needs; raises with what to change."""
    sc, cc, pc, mc = runner.scheduler_config, runner.cache_config, runner.parallel_config, runner.model_config
    bad = []
    if sc.enable_chunked_prefill:
        bad.append("enable_chunked_prefill must be False (--no-enable-chunked-prefill)")
    if getattr(sc, "long_prefill_token_threshold", 0):
        bad.append("long_prefill_token_threshold must be 0")
    if sc.max_num_batched_tokens < mc.max_model_len:
        bad.append(
            f"max_num_batched_tokens ({sc.max_num_batched_tokens}) must be >= max_model_len " f"({mc.max_model_len})"
        )
    if cc.enable_prefix_caching:
        bad.append("enable_prefix_caching must be False (--no-enable-prefix-caching)")
    if runner.speculative_config is not None:
        bad.append("speculative decoding is not supported")
    if getattr(pc, "decode_context_parallel_size", 1) != 1 or getattr(pc, "prefill_context_parallel_size", 1) != 1:
        bad.append("context parallelism (DCP/PCP) is not supported")
    if getattr(pc, "use_ubatching", False):
        bad.append("ubatching (DBO) is not supported")
    if mc.is_encoder_decoder:
        bad.append("encoder-decoder models are not supported")
    if str(cc.cache_dtype) not in ("auto", "bfloat16", "float16"):
        bad.append(f"kv cache dtype {cc.cache_dtype} is not supported (bf16/fp16 only)")
    if getattr(cc, "kv_sharing_fast_prefill", False):
        bad.append("kv_sharing_fast_prefill is not supported")
    if bad:
        raise RuntimeError(
            f"{ENV}: the model folder reads its prompt bidirectionally but the engine settings "
            "do not allow a whole-prompt prefill: " + "; ".join(bad)
        )


def _runner_init(runner, kind: str) -> _RunnerState:
    st = _RunnerState(runner, kind)
    runner._startlux_prefix = st
    with _lock:
        STATUS["runners"] += 1
        STATUS["bi_runners"] += int(st.bi)
    if st.bi:
        if not STATUS["installed"]:
            raise RuntimeError(
                f"{ENV}: {st.model} reads its prompt bidirectionally but the patch is not "
                f"installed ({STATUS['error']})"
            )
        _check_runner_config(runner)
        _say(
            f"{ACTIVE_LINE} for {st.model} (model runner {kind}, chunked prefill off, prefix caching off, "
            f"max_num_batched_tokens {runner.scheduler_config.max_num_batched_tokens}, check={st.check})"
        )
    else:
        _say(f"model {st.model} is causal ({PROMPT_ATTENTION_KEY}={st.mode}): stock attention")
    return st


def _check_backends(runner) -> None:
    """Every full-attention layer must run the Ascend attention backend whose
    metadata is AscendMetadata (the only one the fix-up understands)."""
    if not hasattr(runner, "attn_groups"):
        return  # V2 keeps its groups in the model state; _attach still guards per step
    from vllm.v1.kv_cache_interface import AttentionSpec

    names = set()
    for groups in runner.attn_groups:
        for g in groups:
            if isinstance(g.kv_cache_spec, AttentionSpec):
                names.add(g.backend.get_name())
    if not names or names != {"CUSTOM"}:
        raise RuntimeError(
            f"{ENV}: full-attention layers use {sorted(names)}; the prefix mask needs the "
            "default AscendAttentionBackend (name CUSTOM)"
        )
    _say(f"full-attention backend(s): {sorted(names)}")


def _req_prompt_end(st: _RunnerState, state) -> int:
    p = getattr(state, "_startlux_prompt_end", None)
    if p is not None:
        return p
    ids = state.prompt_token_ids
    if ids is None:
        raise RuntimeError(f"{ENV}: prompt embeddings are not supported (no prompt token ids)")
    n = len(ids)
    sp = state.sampling_params
    extra = (sp.extra_args or {}) if sp is not None else {}
    ex = extra.get(EXTRA_KEY)
    if ex is not None:
        p, kind = int(ex), "explicit"
        if not 0 <= p <= n:
            raise ValueError(f"{ENV}: request {state.req_id}: {EXTRA_KEY}={p} outside [0, {n}]")
    else:
        p, kind = st.prompt_end()(ids)
        if p is None:
            policy = env("_UNMATCHED", "error")
            if policy == "causal":
                p = 0
            elif policy == "full":
                p = n
            else:
                raise RuntimeError(
                    f"{ENV}: request {state.req_id} ({n} prompt tokens) ends neither with "
                    f"{ONE_PASS_SUFFIX!r} nor contains {HEADER + THINK!r} + newline; pass "
                    f"SamplingParams(extra_args={{'{EXTRA_KEY}': p}}) or set "
                    f"{ENV}_UNMATCHED=causal|full"
                )
    state._startlux_prompt_end = p
    with _lock:
        STATUS[kind] += 1
        STATUS["bi_requests"] += 1
        STATUS["max_p"] = max(STATUS["max_p"], p)
    if st.logged < 3:
        st.logged += 1
        _say(f"request {state.req_id}: {n} prompt tokens, prompt end p={p} ({kind})")
    return p


class Plan:
    """The prefix fix-up of one step: flattened-batch rows to recompute, per-request prompt lengths."""

    __slots__ = ("idx", "cu_list", "maxp", "reqs", "ntok", "rows")

    def __repr__(self):
        return f"Plan(reqs={self.reqs[:4]}..., rows={self.rows}, maxp={self.maxp}, ntok={self.ntok})"


def _plan(st: _RunnerState, items, ntok: int) -> Plan | None:
    """items = (batch index, request state, computed tokens, scheduled tokens, first batch row, prompt tokens)."""
    import numpy as np
    import torch

    rows, cu, reqs = [], [0], []
    for i, state, c, s, a0, npt in items:
        p = _req_prompt_end(st, state)
        if p <= 1 or c >= p:
            continue
        if c != 0 or s < p:
            raise RuntimeError(
                f"{ENV}: request {state.req_id}: its prompt (p={p}) would be split across steps "
                f"(computed {c}, scheduled {s}); chunked prefill and prefix caching must be off"
            )
        if s > npt:  # prefill holding generated tokens: a request resumed after preemption
            with _lock:
                STATUS["resumed_prefills"] += 1
        rows.append(np.arange(a0, a0 + p, dtype=np.int64))
        cu.append(cu[-1] + p)
        reqs.append((i, a0, s, p))
    if not reqs:
        return None
    plan = Plan()
    plan.idx = torch.from_numpy(np.concatenate(rows)).to(st.device, non_blocking=True)
    plan.cu_list = cu
    plan.maxp = max(r[3] for r in reqs)
    plan.reqs = reqs
    plan.ntok = ntok
    plan.rows = cu[-1]
    return plan


def _make_plan_v1(runner, num_reqs: int, num_scheduled_tokens: dict) -> Plan | None:
    """NPUModelRunner V1: plan from the persistent input batch + scheduled token counts."""
    import numpy as np

    ib = runner.input_batch
    req_ids = ib.req_ids
    if len(req_ids) < num_reqs:
        raise RuntimeError(f"{ENV}: input batch has {len(req_ids)} requests, step has {num_reqs}")
    nc = ib.num_computed_tokens_cpu[:num_reqs]
    npt = ib.num_prompt_tokens[:num_reqs]
    pre = np.flatnonzero(nc < npt)
    if pre.size == 0:
        return None
    pre_set = set(pre.tolist())
    sched = np.empty(num_reqs, dtype=np.int64)
    for i in range(num_reqs):
        s = num_scheduled_tokens.get(req_ids[i])
        if s is None:
            if i in pre_set:
                raise RuntimeError(f"{ENV}: prefilling request {req_ids[i]} is missing from " "num_scheduled_tokens")
            s = 1
        sched[i] = s
    qs = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(sched, out=qs[1:])
    items = []
    for i in pre.tolist():
        state = runner.requests.get(req_ids[i])
        if state is None:
            # dummy/warm-up rows that are not real requests: skip rather than raise
            continue
        items.append((i, state, int(nc[i]), int(sched[i]), int(qs[i]), int(npt[i])))
    return _plan(runner._startlux_prefix, items, int(qs[-1]))


class _NewRequest:
    """V2: what the plan needs of a request (V2 keeps no per-request object with prompt ids + sampling params)."""

    __slots__ = ("req_id", "prompt_token_ids", "sampling_params", "num_prompt", "_startlux_prompt_end")

    def __init__(self, new_req_data):
        self.req_id = new_req_data.req_id
        self.prompt_token_ids = new_req_data.prompt_token_ids
        self.sampling_params = new_req_data.sampling_params
        n = getattr(new_req_data, "prompt_len", None)
        if n is None:
            ids = new_req_data.prompt_token_ids
            n = len(ids) if ids is not None else 0
        self.num_prompt = n
        self._startlux_prompt_end = None


def _make_plan_v2(runner, ib) -> Plan | None:
    """NPUModelRunner V2: plan from the step's AscendInputBatch (requests in batch order)."""
    if getattr(runner, "_startlux_warmup", False) or getattr(ib, "is_dummy", False):
        with _lock:
            STATUS["warmup_steps"] += 1
        return None
    import numpy as np

    n = ib.num_reqs
    pre = np.flatnonzero(ib.is_prefilling_np[:n])
    if pre.size == 0:
        return None
    known = runner._startlux_reqs
    items = []
    for i in pre.tolist():
        state = known.get(ib.req_ids[i])
        if state is None:
            raise RuntimeError(
                f"{ENV}: request {ib.req_ids[i]} is prefilled in this step but was not seen " "as a new request"
            )
        items.append(
            (
                i,
                state,
                int(ib.num_computed_tokens_np[i]),
                int(ib.num_scheduled_tokens[i]),
                int(ib.query_start_loc_np[i]),
                int(state.num_prompt),
            )
        )
    return _plan(runner._startlux_prefix, items, int(ib.num_tokens))


def _attach(attn_metadata, plan: Plan) -> None:
    """Attach the plan to every AscendMetadata (full-attention layers) of the step."""
    from vllm_ascend.attention.attention_v1 import AscendMetadata

    if isinstance(attn_metadata, list):
        raise RuntimeError(f"{ENV}: ubatching (a list of metadata) is not supported")
    n_full = 0
    for m in {id(m): m for m in attn_metadata.values()}.values():
        if isinstance(m, AscendMetadata):
            if m.num_actual_tokens and m.num_actual_tokens < plan.ntok:
                raise RuntimeError(f"{ENV}: metadata covers {m.num_actual_tokens} tokens, plan {plan.ntok}")
            if m.causal is not True:
                raise RuntimeError(f"{ENV}: the stock attention of this step is already non-causal")
            m.startlux_plan = plan
            n_full += 1
        else:
            # GDN / mamba / linear-attention metadata of hybrid models: keep causal, skip.
            cls = type(m).__name__.lower()
            if any(k in cls for k in ("mamba", "gdn", "linear", "conv", "common")):
                continue
            raise RuntimeError(
                f"{ENV}: attention metadata {type(m).__module__}.{type(m).__name__} is not "
                "supported (need the default AscendAttentionBackend)"
            )
    if n_full == 0:
        raise RuntimeError(f"{ENV}: no AscendMetadata (full-attention) in this step")
    with _lock:
        STATUS["steps_with_plan"] += 1
        STATUS["bi_rows"] += plan.rows


# ----------------------------------------------------------- lm_head letter slice
class _LmHeadProxy:
    """lm_head lookalike whose weight is a row subset of the real head: the
    logits processor's quant_method.apply() only reads .weight/.quant_method,
    so the projection comes out [rows, k] instead of [rows, vocab]."""

    __slots__ = ("_lh", "_w")

    def __init__(self, lh, cols):
        self._lh = lh
        self._w = lh.weight.index_select(0, cols)

    @property
    def weight(self):
        return self._w

    def __getattr__(self, name):
        return getattr(self._lh, name)


def _allowed_union(runner, rows: int):
    """Union of the columns this step's requests allow, or None.

    Engages only when every sampled row of the step belongs to a request that
    restricts its vocab with SamplingParams(allowed_token_ids=...) -- the
    Instant readout contract. Spec-decode rows (rows != num_reqs) and partially
    restricted batches fall back to the stock full-vocab matmul.
    """
    ib = getattr(runner, "input_batch", None)
    if ib is None:
        return None
    n = getattr(ib, "num_reqs", 0)
    if not n or n != rows or len(getattr(ib, "has_allowed_token_ids", ())) != n:
        return None
    mask = getattr(getattr(ib, "sampling_metadata", None), "allowed_token_ids_mask", None)
    if mask is None or mask.dim() != 2 or mask.shape[0] != rows:
        return None
    cols = (~mask).any(0).nonzero(as_tuple=False).squeeze(-1)
    return cols if 0 < cols.numel() < mask.shape[1] else None


def _wrap_lm_head(runner) -> None:
    """compute_logits: project hidden states onto just the union of the
    requests' allowed_token_ids columns, then scatter into the vocab-shaped
    logits tensor as -inf elsewhere. The sampler's allowed_token_ids_mask
    removes those columns anyway, so the post-mask logits are identical to
    stock while the full-vocab matmul is skipped."""
    if env("_LM_HEAD", "1") == "0":
        return
    model = getattr(runner, "model", None)
    orig = getattr(model, "compute_logits", None)
    if model is None or orig is None or getattr(orig, "_startlux_wrapped", False):
        return
    # On *ForConditionalGeneration wrappers the head lives on the inner
    # language model (e.g. AscendQwen3_5ForConditionalGeneration.language_model).
    head = model
    for cand in (model, getattr(model, "language_model", None)):
        lp, lh = getattr(cand, "logits_processor", None), getattr(cand, "lm_head", None)
        if lp is not None and lh is not None:
            head = cand
            break
    else:
        return

    def compute_logits(hidden_states, *args, **kwargs):
        cols = _allowed_union(runner, hidden_states.shape[0])
        if cols is None or getattr(lh, "tp_size", 1) != 1:
            return orig(hidden_states, *args, **kwargs)
        try:
            sub = lp(_LmHeadProxy(lh, cols), hidden_states)
        except Exception:
            return orig(hidden_states, *args, **kwargs)
        if sub is None or sub.dim() != 2 or sub.shape[1] != cols.numel():
            return orig(hidden_states, *args, **kwargs)
        mask = runner.input_batch.sampling_metadata.allowed_token_ids_mask
        logits = sub.new_full((sub.shape[0], mask.shape[1]), float("-inf"))
        logits[:, cols] = sub
        with _lock:
            STATUS["lmhead_calls"] += 1
            STATUS["lmhead_cols"] = cols.numel()
        return logits

    compute_logits._startlux_wrapped = True
    model.compute_logits = compute_logits
    STATUS["patched"].append("lm_head_letters")


# ---------------------------------------------------------------- attention side
def _fixup(impl, plan: Plan, query, key, value, out) -> None:
    """Recompute the prompt rows of the plan non-causally and overwrite them in out.

    query/key/value: [num_tokens, heads, head_size] of this step (the same
    tensors the stock path just wrote into the KV cache). out: the attention
    output the caller reads, [num_tokens, heads*head_size] or [num_tokens,
    heads, head_size].
    """
    import torch_npu

    if key is None or value is None:
        raise RuntimeError(f"{ENV}: a full-attention layer without its own keys/values (KV sharing)")
    if getattr(impl, "alibi_slopes", None) is not None or getattr(impl, "sinks", None) is not None:
        raise RuntimeError(f"{ENV}: ALiBi / attention sinks are not supported")
    if getattr(impl, "sliding_window", None) is not None:
        raise RuntimeError(f"{ENV}: sliding-window full attention is not supported")
    kvd = str(getattr(impl, "kv_cache_dtype", ""))
    if "fp8" in kvd or "float8" in kvd or "int8" in kvd:
        raise RuntimeError(f"{ENV}: quantized KV cache is not supported")

    row_shape = tuple(out.shape[1:])
    kw = dict(head_num=impl.num_heads, input_layout="TND", scale=impl.scale)
    if plan.rows >= int(env("_LEAN_ROWS", str(LEAN_ROWS))) and out.dtype == query.dtype:
        # One non-causal call per request over contiguous in-place views: no
        # [rows, H, D] gather copies (a 128K prompt is several GB per layer).
        for _, a0, _, p in plan.reqs:
            o = torch_npu.npu_fusion_attention(
                query=query[a0 : a0 + p],
                key=key[a0 : a0 + p],
                value=value[a0 : a0 + p],
                actual_seq_qlen=[p],
                actual_seq_kvlen=[p],
                **kw,
            )[0]
            out[a0 : a0 + p] = o.reshape(p, *row_shape)
    else:
        q = query.index_select(0, plan.idx)
        k = key.index_select(0, plan.idx)
        v = value.index_select(0, plan.idx)
        o = torch_npu.npu_fusion_attention(
            query=q, key=k, value=v, actual_seq_qlen=plan.cu_list, actual_seq_kvlen=plan.cu_list, **kw
        )[0]
        out.index_copy_(0, plan.idx, o.reshape(plan.rows, *row_shape).to(out.dtype))
    with _lock:
        STATUS["fixup_calls"] += 1


def _check(impl, plan: Plan, query, key, value, output, stock_rows) -> None:
    """fp32 explicit-mask reference check of one layer call (STARTLUX_PREFIX_BIDIR_CHECK=1)."""
    import torch

    nreq = int(env("_CHECK_REQS", "2"))
    lim = int(env("_CHECK_TOKENS", "8192"))
    g = impl.num_heads // impl.num_kv_heads
    cu = plan.cu_list
    for j, (_, a0, s, p) in enumerate(plan.reqs[:nreq]):
        if s > lim:
            continue
        q = query[a0 : a0 + s].float()
        k = key[a0 : a0 + s].float().repeat_interleave(g, dim=1)
        v = value[a0 : a0 + s].float().repeat_interleave(g, dim=1)
        got = output[a0 : a0 + s].float()
        stock = stock_rows[cu[j] : cu[j + 1]].float() if stock_rows is not None else None
        pos = torch.arange(s, device=q.device)
        e_bi = e_st = e_rest = eff = scale = 0.0
        for b0 in range(0, s, 256):
            qi = pos[b0 : b0 + 256]
            sc = torch.einsum("qhd,khd->hqk", q[b0 : b0 + 256], k) * impl.scale
            causal = pos[None, :] <= qi[:, None]
            mask = causal | ((qi[:, None] < p) & (pos[None, :] < p))
            ref = torch.einsum("hqk,khd->qhd", sc.masked_fill(~mask[None], float("-inf")).softmax(-1), v)
            refc = torch.einsum("hqk,khd->qhd", sc.masked_fill(~causal[None], float("-inf")).softmax(-1), v)
            scale = max(scale, float(ref.abs().max()))
            d = (ref - got[b0 : b0 + 256]).abs().amax(dim=(1, 2))
            inp = qi < p
            if bool(inp.any()):
                n_in = int(inp.sum())
                e_bi = max(e_bi, float(d[inp].max()))
                if stock is not None:
                    e_st = max(e_st, float((refc[:n_in] - stock[b0 : b0 + n_in]).abs().max()))
                eff = max(eff, float((ref - refc)[inp].abs().max()))
            if bool((~inp).any()):
                e_rest = max(e_rest, float(d[~inp].max()))
        lim_err = max(2.0 * e_st, scale * 2.0**-7) + 1e-3
        with _lock:
            STATUS["check_calls"] += 1
            STATUS["check_max_diff_prompt"] = max(STATUS["check_max_diff_prompt"], e_bi)
            STATUS["check_max_diff_rest"] = max(STATUS["check_max_diff_rest"], e_rest)
            STATUS["check_max_stock_err"] = max(STATUS["check_max_stock_err"], e_st)
            if stock is not None:
                STATUS["check_max_ratio"] = max(STATUS["check_max_ratio"], e_bi / max(e_st, 1e-6))
            if p > 1:
                m = STATUS["check_min_bidir_effect"]
                STATUS["check_min_bidir_effect"] = eff if m is None else min(m, eff)
        if STATUS["check_calls"] <= 8 or e_bi > lim_err or e_rest > lim_err:
            _say(
                f"check: rows [{a0}, {a0 + s}) p={p}: |fixup - fp32 prefix ref| {e_bi:.4g}, |stock causal - fp32 "
                f"causal ref| {e_st:.4g}, |rest - ref| {e_rest:.4g}, bidirectional effect {eff:.4g}, max|ref| "
                f"{scale:.4g}"
            )
        if e_bi > lim_err or e_rest > lim_err or (p > 1 and eff == 0.0):
            raise RuntimeError(
                f"{ENV} check failed: rows [{a0}, {a0 + s}) p={p}: fixup err {e_bi:.4g}, rest "
                f"err {e_rest:.4g}, stock causal err {e_st:.4g}, effect {eff:.4g}, limit {lim_err:.4g}"
            )


def _patch_ascend_attn(mod) -> None:
    """vllm_ascend.attention.attention_v1: run the plan fix-up after the stock causal call."""
    Impl = mod.AscendAttentionBackendImpl
    if getattr(Impl, "_startlux_patched", False):
        return
    fwd0 = Impl.forward
    from vllm.v1.attention.backend import AttentionType

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        plan = getattr(attn_metadata, "startlux_plan", None) if attn_metadata is not None else None
        if plan is not None and self.attn_type == AttentionType.DECODER:
            if _EXTRA_CTX_CAPTURING():
                raise RuntimeError(f"{ENV}: a step needing the prefix fix-up ran inside graph capture")
            if plan.rows == plan.ntok and all(s == p for _, _, s, p in plan.reqs):
                # Pure bidirectional step (Instant one-pass): the stock FIA call
                # itself runs non-causal (sparse_mode=0), so the prompt is
                # computed once instead of causal + fix-up. KV-cache writes
                # inside fwd0 are unaffected by the causal flag.
                attn_metadata.causal = False
                try:
                    out = fwd0(
                        self,
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output=output,
                        output_scale=output_scale,
                        output_block_scale=output_block_scale,
                    )
                finally:
                    attn_metadata.causal = True
                dst = out if out is not None else output
                if dst is None:
                    raise RuntimeError(f"{ENV}: the attention impl returned no output buffer")
                if env("_CHECK") == "1":
                    _check(self, plan, query, key, value, dst, None)
                with _lock:
                    STATUS["direct_calls"] += 1
                return out
        out = fwd0(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
        if plan is not None and self.attn_type == AttentionType.DECODER:
            dst = out if out is not None else output
            if dst is None:
                raise RuntimeError(f"{ENV}: the attention impl returned no output buffer to fix up")
            chk = env("_CHECK") == "1"
            stock_rows = dst.index_select(0, plan.idx).clone() if chk else None
            _fixup(self, plan, query, key, value, dst)
            if chk:
                _check(self, plan, query, key, value, dst, stock_rows)
        return out

    Impl.forward = forward
    Impl._startlux_patched = True
    STATUS["patched"].append("AscendAttentionBackendImpl")

    # C8 (int8 KV) must never see a plan: the engine-config check refuses
    # non-bf16/fp16 caches, but guard anyway so a misconfiguration is loud.
    C8 = getattr(mod, "AscendC8AttentionBackendImpl", None)
    if C8 is not None and not getattr(C8, "_startlux_guarded", False):
        c8_fwd0 = C8.forward

        def c8_forward(self, *args, **kwargs):
            attn_metadata = kwargs.get("attn_metadata", args[5] if len(args) > 5 else None)
            if getattr(attn_metadata, "startlux_plan", None) is not None:
                raise RuntimeError(f"{ENV}: int8 KV cache (C8) is not supported for bidirectional prompts")
            return c8_fwd0(self, *args, **kwargs)

        C8.forward = c8_forward
        C8._startlux_guarded = True
        STATUS["patched"].append("AscendC8AttentionBackendImpl")


def _EXTRA_CTX_CAPTURING() -> bool:
    try:
        from vllm_ascend.attention.attention_v1 import _EXTRA_CTX

        return bool(getattr(_EXTRA_CTX, "capturing", False))
    except Exception:
        return False


# ------------------------------------------------------------------ runner V1
def _patch_runner_v1(mod) -> None:
    """vllm_ascend.worker.model_runner_v1.NPUModelRunner."""
    R = mod.NPUModelRunner
    if getattr(R, "_startlux_patched", False):
        return
    init0, build0, backend0, load0 = (
        R.__init__,
        R._build_attention_metadata,
        R.initialize_attn_backend,
        R.load_model,
    )
    sig = inspect.signature(build0)

    def __init__(self, *args, **kwargs):
        init0(self, *args, **kwargs)
        _runner_init(self, "V1")

    def load_model(self, *args, **kwargs):
        out = load0(self, *args, **kwargs)
        _wrap_lm_head(self)
        return out

    def initialize_attn_backend(self, kv_cache_config, *args, **kwargs):
        backend0(self, kv_cache_config, *args, **kwargs)
        st = getattr(self, "_startlux_prefix", None)
        if st is None or not st.bi:
            return
        _check_backends(self)

    def _build_attention_metadata(self, *args, **kwargs):
        out = build0(self, *args, **kwargs)
        st = getattr(self, "_startlux_prefix", None)
        if st is None or not st.bi:
            return out
        b = sig.bind(self, *args, **kwargs)
        b.apply_defaults()
        a = b.arguments
        if (
            a["for_cudagraph_capture"]
            or a.get("offload_dummy")
            or a["num_scheduled_tokens"] is None
            or a["max_query_len"] <= 1
        ):
            return out  # capture / dummy runs, steps without a multi-token prefill
        plan = _make_plan_v1(self, a["num_reqs"], a["num_scheduled_tokens"])
        if plan is not None:
            mode = a.get("cudagraph_runtime_mode")
            if getattr(mode, "name", mode) == "FULL":
                raise RuntimeError(
                    f"{ENV}: a step with a prompt prefill was dispatched to a FULL ACL graph; "
                    "the fix-up runs only in eager attention"
                )
            _attach(out[0], plan)
        return out

    R.__init__ = __init__
    R.load_model = load_model
    R.initialize_attn_backend = initialize_attn_backend
    R._build_attention_metadata = _build_attention_metadata
    R._startlux_patched = True
    STATUS["patched"].append("NPUModelRunnerV1")


# ------------------------------------------------------------------ runner V2
def _wrap_prepare_attn(runner) -> None:
    """Wrap the ascend model state's prepare_attn so the plan lands on the
    AscendMetadata it returns."""
    ms = runner.model_state
    prep0 = ms.prepare_attn
    if getattr(prep0, "_startlux_wrapped", False):
        return
    sig = inspect.signature(prep0)

    def prepare_attn(input_batch, cudagraph_mode, *args, **kwargs):
        out = prep0(input_batch, cudagraph_mode, *args, **kwargs)
        b = sig.bind(input_batch, cudagraph_mode, *args, **kwargs)
        b.apply_defaults()
        if b.arguments.get("for_capture") or getattr(input_batch, "is_dummy", False):
            return out
        plan = _make_plan_v2(runner, input_batch)
        if plan is not None:
            if getattr(cudagraph_mode, "name", cudagraph_mode) == "FULL":
                raise RuntimeError(
                    f"{ENV}: a step with a prompt prefill was dispatched to a FULL ACL graph; "
                    "the fix-up runs only in eager attention"
                )
            _attach(out, plan)
        return out

    prepare_attn._startlux_wrapped = True
    ms.prepare_attn = prepare_attn


def _patch_runner_v2(mod) -> None:
    """vllm_ascend.worker.v2.model_runner.NPUModelRunner (upstream V2 GPUModelRunner subclass)."""
    R = mod.NPUModelRunner
    if getattr(R, "_startlux_patched", False):
        return
    init0, load0 = R.__init__, R.load_model
    kv0 = getattr(R, "initialize_kv_cache", None)
    add0, rm0, exec0 = R.add_requests, R._remove_request, R.execute_model

    def bi(runner) -> bool:
        st = getattr(runner, "_startlux_prefix", None)
        return st is not None and st.bi

    def __init__(self, *args, **kwargs):
        init0(self, *args, **kwargs)
        self._startlux_reqs = {}
        _runner_init(self, "V2")

    def load_model(self, *args, **kwargs):
        out = load0(self, *args, **kwargs)
        _wrap_lm_head(self)
        if bi(self):
            _wrap_prepare_attn(self)
        return out

    def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        out = kv0(self, kv_cache_config, *args, **kwargs)
        if bi(self):
            _check_backends(self)
        return out

    def add_requests(self, scheduler_output):
        out = add0(self, scheduler_output)
        if bi(self):
            for nr in scheduler_output.scheduled_new_reqs:
                self._startlux_reqs[nr.req_id] = _NewRequest(nr)
        return out

    def _remove_request(self, req_id):
        reqs = getattr(self, "_startlux_reqs", None)
        if reqs:
            reqs.pop(req_id, None)
        return rm0(self, req_id)

    def execute_model(self, *args, **kwargs):
        if bi(self) and not getattr(self.model_state.prepare_attn, "_startlux_wrapped", False):
            raise RuntimeError(
                f"{ENV}: the model state of {self._startlux_prefix.model} was replaced after "
                "load_model; its attention metadata would not get the prefix plan"
            )
        return exec0(self, *args, **kwargs)

    R.__init__ = __init__
    R.load_model = load_model
    if kv0 is not None:
        R.initialize_kv_cache = initialize_kv_cache
    R.add_requests = add_requests
    R._remove_request = _remove_request
    R.execute_model = execute_model
    R._startlux_patched = True
    STATUS["patched"].append("NPUModelRunnerV2")


# ------------------------------------------------------------------ worker side
def _status(worker) -> dict:
    mr = getattr(worker, "model_runner", None)
    st = getattr(mr, "_startlux_prefix", None)
    with _lock:
        d = json.loads(json.dumps(STATUS))
    d.update(
        {
            "pid": os.getpid(),
            "model": st.model if st else None,
            "mode": st.mode if st else None,
            "runner": f"{type(mr).__module__}.{type(mr).__name__}" if mr is not None else None,
            "active": bool(st is not None and st.bi and STATUS["installed"]),
            "env": env() or os.environ.get(ENV),
            "check": env("_CHECK"),
        }
    )
    return d


def _require_patched_runner(worker) -> None:
    """A bidirectional folder must not run on a runner this module did not patch."""
    model = str(worker.model_config.model)
    if not (os.path.isdir(model) and read_prompt_attention(model) == "bidirectional"):
        return
    mr = getattr(worker, "model_runner", None)
    st = getattr(mr, "_startlux_prefix", None)
    if st is None or not st.bi:
        why = STATUS["error"] or "this runner is not supported"
        raise RuntimeError(
            f"{ENV}: {model} reads its prompt bidirectionally but its model runner "
            f"{type(mr).__module__}.{type(mr).__name__} is not patched ({why}); "
            "the NPU would read the folder causally"
        )


def _patch_worker(mod) -> None:
    W = mod.NPUWorker
    if not hasattr(W, "startlux_prefix_status"):
        W.startlux_prefix_status = _status
        STATUS["patched"].append("NPUWorker.startlux_prefix_status")
    if not getattr(W, "_startlux_guarded", False):
        init_device0 = W.init_device

        def init_device(self, *args, **kwargs):
            out = init_device0(self, *args, **kwargs)
            _require_patched_runner(self)
            return out

        W.init_device = init_device
        W._startlux_guarded = True
        STATUS["patched"].append("NPUWorker.init_device")


def _patch_warmup(mod) -> None:
    """vllm_ascend.model_executor.warmup.kernel_warmup: while it runs, its worker's
    model runner makes no plans (synthetic prompts are read causally)."""
    fn = getattr(mod, "kernel_warmup", None)
    if fn is None or getattr(fn, "_startlux_wrapped", False):
        return

    @functools.wraps(fn)
    def wrapper(worker, *args, **kwargs):
        mr = getattr(worker, "model_runner", None)
        prev = getattr(mr, "_startlux_warmup", False)
        if mr is not None:
            mr._startlux_warmup = True
        try:
            return fn(worker, *args, **kwargs)
        finally:
            if mr is not None:
                mr._startlux_warmup = prev

    mod.kernel_warmup = wrapper
    STATUS["patched"].append("kernel_warmup")


# ------------------------------------------------------------------ installation
_HOOKS: dict = {}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs a patch function right after a target module is executed."""

    def find_spec(self, name, path=None, target=None):
        if name not in _HOOKS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        fn = _HOOKS[name]

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec_):
                return loader.create_module(spec_)

            def exec_module(self, module):
                loader.exec_module(module)
                fn(module)

        spec.loader = _Loader()
        return spec


def _hook(name: str, fn) -> None:
    _HOOKS[name] = fn
    if name in sys.modules:
        fn(sys.modules[name])
    STATUS["hooked"].append(name)


def install() -> bool:
    """Install the import hooks (idempotent). Called at vllm_ascend import when
    STARTLUX_PREFIX_BIDIR=1 or VLLM_ASCEND_STARTLUX_PREFIX_BIDIR=1."""
    if STATUS["installed"] or STATUS["hooked"]:
        return STATUS["installed"]
    sys.meta_path.insert(0, _PostImportFinder())
    STATUS["installed"] = True
    _hook("vllm_ascend.worker.model_runner_v1", _patch_runner_v1)
    _hook("vllm_ascend.worker.v2.model_runner", _patch_runner_v2)
    _hook("vllm_ascend.attention.attention_v1", _patch_ascend_attn)
    _hook("vllm_ascend.worker.worker", _patch_worker)
    _hook("vllm_ascend.model_executor.warmup.kernel_warmup", _patch_warmup)
    _say(f"installed (hooks: {', '.join(STATUS['hooked'])})")
    return True


def plugin() -> None:
    if enabled():
        install()


def status(llm) -> list:
    try:
        return llm.collective_rpc("startlux_prefix_status")
    except Exception as e:
        raise RuntimeError(
            f"{ENV} is not loaded in the engine ({type(e).__name__}: {e}); set " f"{ENV}=1 or {ENV_ASCEND}=1"
        ) from e


def require_active(llm, model_dir: str) -> list:
    sts = status(llm) if read_prompt_attention(model_dir) == "bidirectional" else []
    for s in sts:
        if not s.get("active"):
            raise RuntimeError(f"{ENV}: {model_dir} is bidirectional but an engine worker is not patched: {s}")
    return sts
