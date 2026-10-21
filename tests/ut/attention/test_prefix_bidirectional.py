# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for vllm_ascend.attention.prefix_bidirectional.

The NPU kernels are not exercised here; this covers the prompt-end detection,
plan row math, metadata attachment rules and engine-config validation that run
on the host side.
"""

import json
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.attention.prefix_bidirectional as pb
from vllm_ascend.attention.attention_v1 import AscendMetadata

THINK_ID = 200
NL_ID = 198


class FakeTokenizer:
    """Tokenizes '<think>' atomically (id 200), '\\n' atomically (id 198), every
    other character to its ord()."""

    def encode(self, s, add_special_tokens=False):
        ids, i = [], 0
        while i < len(s):
            if s.startswith("<think>", i):
                ids.append(THINK_ID)
                i += len("<think>")
            else:
                ids.append(ord(s[i]))
                i += 1
        return ids

    def convert_tokens_to_ids(self, tok):
        return {"<think>": THINK_ID}.get(tok, -1)


@pytest.fixture(scope="module")
def prompt_end():
    return pb.PromptEnd(FakeTokenizer())


def _ids(s: str):
    return FakeTokenizer().encode(s)


def _req(req_id, ids, extra=None):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=ids,
        sampling_params=SimpleNamespace(extra_args=extra),
        _startlux_prompt_end=None,
    )


def _runner_state(device="cpu"):
    st = SimpleNamespace(device=device, logged=0, check=False)
    st.prompt_end = lambda: pb.PromptEnd(FakeTokenizer())
    return st


class TestPromptEnd:
    def test_one_pass_suffix(self, prompt_end):
        ids = _ids("user question body<|im_start|>assistant\n") + _ids("<think>\n\n</think>\n\n")
        p, kind = prompt_end(ids)
        assert kind == "one_pass"
        assert p == len(ids)

    def test_think_header(self, prompt_end):
        prefix = _ids("context<|im_start|>assistant\n<think>\n")
        ids = prefix + _ids("the model keeps thinking here")
        p, kind = prompt_end(ids)
        assert kind == "think"
        assert p == len(prefix)

    def test_think_requires_newline(self, prompt_end):
        ids = _ids("<|im_start|>assistant\n<think>") + _ids("no newline")
        p, kind = prompt_end(ids)
        assert (p, kind) == (None, "unmatched")

    def test_unmatched(self, prompt_end):
        p, kind = prompt_end(_ids("just some prompt"))
        assert (p, kind) == (None, "unmatched")


class TestReadPromptAttention:
    def test_bidirectional(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"jev_prompt_attention": "bidirectional"}))
        assert pb.read_prompt_attention(str(tmp_path)) == "bidirectional"

    def test_absent_means_causal(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({}))
        assert pb.read_prompt_attention(str(tmp_path)) == "causal"

    def test_missing_file_means_causal(self, tmp_path):
        assert pb.read_prompt_attention(str(tmp_path / "nope")) == "causal"


class TestReqPromptEnd:
    def test_explicit_extra_args(self):
        st = _runner_state()
        ids = _ids("anything at all")
        req = _req("r1", ids, extra={pb.EXTRA_KEY: 3})
        assert pb._req_prompt_end(st, req) == 3

    def test_explicit_out_of_range_raises(self):
        st = _runner_state()
        req = _req("r1", _ids("short"), extra={pb.EXTRA_KEY: 999})
        with pytest.raises(ValueError, match="outside"):
            pb._req_prompt_end(st, req)

    def test_unmatched_raises_by_default(self, monkeypatch):
        monkeypatch.delenv(pb.ENV + "_UNMATCHED", raising=False)
        req = _req("r1", _ids("no think here"))
        with pytest.raises(RuntimeError, match="extra_args"):
            pb._req_prompt_end(_runner_state(), req)

    def test_unmatched_causal_and_full(self, monkeypatch):
        ids = _ids("no think here")
        monkeypatch.setenv(pb.ENV + "_UNMATCHED", "causal")
        assert pb._req_prompt_end(_runner_state(), _req("r1", ids)) == 0
        monkeypatch.setenv(pb.ENV + "_UNMATCHED", "full")
        assert pb._req_prompt_end(_runner_state(), _req("r1", ids)) == len(ids)

    def test_memoized(self):
        st = _runner_state()
        ids = _ids("x<|im_start|>assistant\n<think>\nrest")
        req = _req("r1", ids)
        p1 = pb._req_prompt_end(st, req)
        req.prompt_token_ids = _ids("mutated")
        assert pb._req_prompt_end(st, req) == p1


class TestPlan:
    def _items(self, ids_a, ids_b, a0, b0):
        ra, rb = _req("a", ids_a), _req("b", ids_b)
        return [(0, ra, 0, len(ids_a), a0, len(ids_a)), (1, rb, 0, len(ids_b), b0, len(ids_b))]

    def test_rows_and_cu(self):
        st = _runner_state()
        suffix = _ids("<|im_start|>assistant\n<think>\n")
        ids_a = _ids("AAAA") + suffix + _ids("tail")  # p = 4 + len(suffix)
        ids_b = ids_a
        items = self._items(ids_a, ids_b, a0=0, b0=len(ids_a))
        plan = pb._plan(st, items, ntok=2 * len(ids_a))
        p = 4 + len(suffix)
        assert plan.reqs == [(0, 0, len(ids_a), p), (1, len(ids_a), len(ids_a), p)]
        assert plan.cu_list == [0, p, 2 * p]
        assert plan.rows == 2 * p
        assert plan.idx.tolist() == list(range(p)) + list(range(len(ids_a), len(ids_a) + p))
        assert plan.maxp == p

    def test_no_prefills_returns_none(self):
        st = _runner_state()
        ids = _ids("x<|im_start|>assistant\n<think>\nz")
        items = [(0, _req("a", ids), len(ids), 1, 0, len(ids))]
        assert pb._plan(st, items, ntok=1) is None

    def test_split_prompt_raises(self):
        st = _runner_state()
        ids = _ids("x<|im_start|>assistant\n<think>\nzzzz")
        p = pb._req_prompt_end(st, _req("probe", ids))
        items = [(0, _req("a", ids), 0, p - 1, 0, len(ids))]  # scheduled < p
        with pytest.raises(RuntimeError, match="split across steps"):
            pb._plan(st, items, ntok=p - 1)

    def test_computed_beyond_p_skipped(self):
        st = _runner_state()
        ids = _ids("x<|im_start|>assistant\n<think>\nzzzz")
        p = pb._req_prompt_end(st, _req("probe", ids))
        items = [(0, _req("a", ids), p, 4, 0, len(ids))]
        assert pb._plan(st, items, ntok=4) is None


class FakeGDNAttnMetadata:
    pass


class WeirdOtherMetadata:
    pass


class TestAttach:
    def _plan(self):
        plan = pb.Plan()
        plan.ntok = 10
        plan.rows = 4
        plan.maxp = 4
        plan.reqs = [(0, 0, 10, 4)]
        plan.cu_list = [0, 4]
        plan.idx = torch.arange(4)
        return plan

    def test_attaches_to_ascend_metadata_only(self):
        meta = AscendMetadata()
        meta.num_actual_tokens = 10
        d = {"full.0": meta, "gdn.0": FakeGDNAttnMetadata()}
        pb._attach(d, self._plan())
        assert meta.startlux_plan is not None

    def test_list_metadata_raises(self):
        with pytest.raises(RuntimeError, match="ubatching"):
            pb._attach([AscendMetadata()], self._plan())

    def test_unknown_metadata_raises(self):
        with pytest.raises(RuntimeError, match="not supported"):
            pb._attach({"x": WeirdOtherMetadata(), "f": AscendMetadata()}, self._plan())

    def test_non_causal_stock_raises(self):
        meta = AscendMetadata()
        meta.causal = False
        meta.num_actual_tokens = 10
        with pytest.raises(RuntimeError, match="already non-causal"):
            pb._attach({"f": meta}, self._plan())


class TestRunnerConfig:
    def _runner(self):
        sc = SimpleNamespace(
            enable_chunked_prefill=False, long_prefill_token_threshold=0, max_num_batched_tokens=140000
        )
        cc = SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto", kv_sharing_fast_prefill=False)
        pc = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1, use_ubatching=False)
        mc = SimpleNamespace(max_model_len=131072, is_encoder_decoder=False)
        return SimpleNamespace(
            scheduler_config=sc, cache_config=cc, parallel_config=pc, model_config=mc, speculative_config=None
        )

    def test_good_config_passes(self):
        pb._check_runner_config(self._runner())

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("scheduler_config.enable_chunked_prefill", True),
            ("scheduler_config.long_prefill_token_threshold", 1024),
            ("scheduler_config.max_num_batched_tokens", 8192),
            ("cache_config.enable_prefix_caching", True),
            ("cache_config.cache_dtype", "fp8"),
            ("parallel_config.decode_context_parallel_size", 2),
            ("parallel_config.use_ubatching", True),
            ("model_config.is_encoder_decoder", True),
            ("speculative_config", object()),
        ],
    )
    def test_bad_config_raises(self, field, bad):
        runner = self._runner()
        obj, attr = runner, field
        if "." in field:
            obj_path, attr = field.rsplit(".", 1)
            obj = runner
            for part in obj_path.split("."):
                obj = getattr(obj, part)
        setattr(obj, attr, bad)
        with pytest.raises(RuntimeError, match="bidirectionally"):
            pb._check_runner_config(runner)


class TestEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(pb.ENV, raising=False)
        monkeypatch.delenv("VLLM_ASCEND_STARTLUX_PREFIX_BIDIR", raising=False)
        assert pb.enabled() is False

    def test_upstream_flag(self, monkeypatch):
        monkeypatch.setenv(pb.ENV, "1")
        assert pb.enabled() is True


class TestDirectPath:
    """Pure bidirectional steps flip the stock FIA call to non-causal once
    instead of running causal + fix-up (direct one-pass)."""

    class _Impl:
        def __init__(self):
            from vllm.v1.attention.backend import AttentionType

            self.attn_type = AttentionType.DECODER
            self.seen_causal = []

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, **kw):
            self.seen_causal.append(attn_metadata.causal)
            return output

    def _wrap(self):
        mod = SimpleNamespace(AscendAttentionBackendImpl=self._Impl)
        pb._patch_ascend_attn(mod)
        impl = self._Impl()
        return mod, impl

    @staticmethod
    def _meta(plan, ntok):
        meta = AscendMetadata()
        meta.causal = True
        meta.num_actual_tokens = ntok
        meta.startlux_plan = plan
        return meta

    @staticmethod
    def _plan(reqs, ntok, rows):
        plan = pb.Plan()
        plan.reqs = reqs
        plan.ntok = ntok
        plan.rows = rows
        plan.cu_list = [0, rows]
        plan.idx = torch.arange(rows)
        plan.maxp = max(p for _, _, _, p in reqs)
        return plan

    def test_pure_step_runs_stock_non_causal(self):
        _, impl = self._wrap()
        plan = self._plan([(0, 0, 10, 10)], ntok=10, rows=10)
        meta = self._meta(plan, ntok=10)
        out_buf = torch.zeros(10, 8)
        q = torch.randn(10, 2, 4)
        before = pb.STATUS["direct_calls"]
        impl.forward(None, q, q, q, None, meta, output=out_buf)
        assert impl.seen_causal == [False]
        assert meta.causal is True  # restored
        assert pb.STATUS["direct_calls"] == before + 1

    def test_mixed_step_keeps_causal_and_fixup(self, monkeypatch):
        _, impl = self._wrap()
        # request scheduled 10 tokens, only the first 4 are bidir prefix rows
        plan = self._plan([(0, 0, 10, 4)], ntok=10, rows=4)
        meta = self._meta(plan, ntok=10)
        out_buf = torch.zeros(10, 2, 4)
        q = torch.randn(10, 2, 4)
        impl.num_heads, impl.num_kv_heads, impl.scale = 2, 2, 0.5
        impl.alibi_slopes = impl.sinks = impl.sliding_window = impl.kv_cache_dtype = None
        fixed = []
        monkeypatch.setattr(pb, "_fixup", lambda *a: fixed.append(a))
        impl.forward(None, q, q, q, None, meta, output=out_buf)
        assert impl.seen_causal == [True]
        assert len(fixed) == 1


class TestLmHeadSlice:
    """_allowed_union + the compute_logits column-subset path."""

    V = 12
    H = 4

    def _runner(self, masks_per_req, n_reqs=None):
        n_reqs = n_reqs if n_reqs is not None else len(masks_per_req)
        mask = torch.ones(n_reqs, self.V, dtype=torch.bool)
        for i, allowed in enumerate(masks_per_req):
            mask[i, allowed] = False
        ib = SimpleNamespace(
            num_reqs=n_reqs,
            has_allowed_token_ids={f"r{i}" for i in range(len(masks_per_req))},
            sampling_metadata=SimpleNamespace(allowed_token_ids_mask=mask),
        )
        w = torch.randn(self.V, self.H)

        def lp(lm_head, hidden_states):
            return hidden_states @ lm_head.weight.t()

        model = SimpleNamespace(
            logits_processor=lp,
            lm_head=SimpleNamespace(weight=w, tp_size=1),
        )
        model.compute_logits = lambda h, *a, **k: h @ w.t()
        runner = SimpleNamespace(input_batch=ib, model=model)
        pb._wrap_lm_head(runner)
        return runner, w, mask

    def test_proxy_weight_subset(self):
        lh = SimpleNamespace(weight=torch.randn(20, 4), tp_size=1, tag="orig")
        cols = torch.tensor([2, 5, 9])
        proxy = pb._LmHeadProxy(lh, cols)
        assert torch.equal(proxy.weight, lh.weight[cols])
        assert proxy.tag == "orig" and proxy.tp_size == 1

    def test_no_restriction_stock(self):
        ib = SimpleNamespace(num_reqs=2, has_allowed_token_ids=set(), sampling_metadata=SimpleNamespace())
        runner = SimpleNamespace(
            input_batch=ib, model=SimpleNamespace(compute_logits=lambda h: h, logits_processor=None, lm_head=None)
        )
        pb._wrap_lm_head(runner)
        assert pb._allowed_union(runner, 2) is None

    def test_partial_restriction_stock(self):
        runner, w, _ = self._runner([[1, 2, 3], [4, 5]])
        runner.input_batch.has_allowed_token_ids = {"r0"}  # only one req restricted
        hidden = torch.randn(2, self.H)
        out = runner.model.compute_logits(hidden)
        assert torch.equal(out, hidden @ w.t())

    def test_all_restricted_sliced_and_equivalent(self):
        runner, w, mask = self._runner([[1, 2, 3], [3, 4, 5]])
        hidden = torch.randn(2, self.H)
        before = pb.STATUS["lmhead_calls"]
        out = runner.model.compute_logits(hidden)
        union = [1, 2, 3, 4, 5]
        for c in union:
            assert torch.equal(out[:, c], (hidden @ w.t())[:, c])
        assert bool((out[:, [0, 6, 7, 8, 9, 10, 11]] == float("-inf")).all())
        assert pb.STATUS["lmhead_calls"] == before + 1
        assert pb.STATUS["lmhead_cols"] == len(union)

    def test_full_vocab_union_falls_back(self):
        runner, w, _ = self._runner([[0, 1, 2], list(range(3, self.V))])
        hidden = torch.randn(2, self.H)
        before = pb.STATUS["lmhead_calls"]
        out = runner.model.compute_logits(hidden)
        assert torch.equal(out, hidden @ w.t())
        assert pb.STATUS["lmhead_calls"] == before

    def test_rows_mismatch_stock(self):
        runner, w, _ = self._runner([[1, 2], [3, 4]])
        hidden = torch.randn(3, self.H)  # spec/draft rows -> no 1:1 mapping
        out = runner.model.compute_logits(hidden)
        assert torch.equal(out, hidden @ w.t())
