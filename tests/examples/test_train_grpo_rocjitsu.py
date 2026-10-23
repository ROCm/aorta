"""The trainer's rocjitsu seam: the system prompt, the slices, the refusals.

The silent failure this guards against is a gradient taken on a prompt the
reply was not sampled under: generation and the loss each template the user
turn, and if one of them used the probe environment's system prompt the update
would still run, still pass every weight check, and train the wrong thing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "examples"))

import episode_env  # noqa: E402
import rocjitsu_env as rj  # noqa: E402
import train_grpo_step as trainer  # noqa: E402
from test_rocjitsu_env import ANSWER, build_corpus  # noqa: E402
from test_train_grpo_step import (  # noqa: E402
    CharTokenizer,
    TinyLM,
    _GenArgs,
    _PadTokenizer,
    _SeedRecorder,
)


@pytest.fixture(autouse=True)
def _default_reward_and_prompt(monkeypatch):
    """The synthetic corpus is v1-layout, which takes reward v1 only; an exported choice must not change that."""
    monkeypatch.delenv(rj.REWARD_ENV, raising=False)
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)


class RecordingTokenizer(CharTokenizer):
    """Records the system message of every template call."""

    def __init__(self) -> None:
        super().__init__()
        self.systems: list[str] = []

    def apply_chat_template(self, messages, **kwargs):
        self.systems.append(next(m["content"] for m in messages if m["role"] == "system"))
        return super().apply_chat_template(messages, **kwargs)


@pytest.fixture()
def scenarios(tmp_path):
    return rj.load_corpus(build_corpus(tmp_path), split="train")


def test_generation_and_the_loss_template_under_the_environments_system_prompt(scenarios, monkeypatch):
    system = rj.system_for(scenarios)
    assert system != episode_env.SYSTEM

    def fake_generate(model, tok, prompts, args):
        return [json.dumps(ANSWER)] * len(prompts)

    monkeypatch.setattr(trainer, "generate", fake_generate)

    class Args:
        group = 2
        log_episodes = 0

    tok = RecordingTokenizer()
    groups, samples, _ = trainer.episode_rollouts(None, tok, scenarios, Args(), rj.TurnBudget(),
                                                  env=rj, system=system)
    assert tok.systems and set(tok.systems) == {system}, "generation"
    assert {g["scenario_id"] for g in groups} == {s.scenario_id for s in scenarios}
    tok.systems.clear()
    sample = samples[0]
    loss = trainer.sample_loss(TinyLM(), tok, sample.__class__(**{**sample.__dict__, "advantage": 1.0}), 1,
                               device="cpu", system=system)
    assert loss is not None and tok.systems == [system], "loss"


class KeepingLM(TinyLM):
    """TinyLM with a ``logits_to_keep`` forward, as transformers' causal LMs have."""

    def __init__(self) -> None:
        super().__init__()
        self.kept: list[int] = []

    def forward(self, input_ids, logits_to_keep=0):
        out = super().forward(input_ids)
        self.kept.append(logits_to_keep)
        if logits_to_keep:
            out.logits = out.logits[:, -logits_to_keep:, :]
        return out


def test_completion_logprobs_keep_only_completion_logits_and_match_the_full_forward():
    ids = torch.tensor([[3, 9, 1, 40, 7, 22, 5, 61]])
    full = torch.log_softmax(TinyLM()(ids).logits[:, :-1, :], dim=-1).gather(
        2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -3:]
    keeping = KeepingLM()
    kept = trainer.completion_logprobs(keeping, ids, 3)
    assert keeping.kept == [4], "the forward is asked for the completion positions plus one"
    assert torch.allclose(kept, full)
    assert torch.allclose(trainer.completion_logprobs(TinyLM(), ids, 3), full), "fallback for a plain forward"


def test_sample_loss_reports_the_worst_token_and_the_sequence_length(scenarios):
    tok = CharTokenizer()
    sample = episode_env.Sample(scenario_id="s", prompt="hi", completion="ok", advantage=1.0)
    _, stats = trainer.sample_loss(TinyLM(), tok, sample, 1, device="cpu")
    n_prompt = tok(trainer.chat_prompt(tok, "hi"), return_tensors="pt")["input_ids"].shape[1]
    assert stats["seq_tokens"] == n_prompt + len(tok("ok", add_special_tokens=False)["input_ids"][0])
    assert 0.0 < stats["max_token_nll"] <= stats["nll"]
    ids = tok(trainer.chat_prompt(tok, "hi") + "ok", return_tensors="pt")["input_ids"]
    with trainer.policy_numerics("cpu"):
        per_token = -trainer.completion_logprobs(TinyLM(), ids, 2)
    assert per_token.max() > per_token.min(), "the fixture must tell max from mean"
    assert stats["max_token_nll"] == pytest.approx(float(per_token.max()))


def test_a_loss_forward_that_disowns_its_own_samples_gates_the_step():
    assert trainer.LOSS_FORWARD_CHECK not in trainer.GATING_CHECKS, "decided before the step"
    assert trainer.loss_forward_check(trainer.MAX_SAMPLED_TOKEN_NLL, 10)["passed"]
    bad = trainer.loss_forward_check(trainer.MAX_SAMPLED_TOKEN_NLL + 1e-3, 8120)
    assert not bad["passed"] and bad["seq_tokens"] == 8120


def _numerics() -> tuple[bool, Any]:
    return torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")


class NumericsLM(TinyLM):
    """Records the autocast state every forward ran under."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[bool, Any]] = []

    def forward(self, input_ids, logits_to_keep=0):
        self.seen.append(_numerics())
        return super().forward(input_ids)


class NumericsGenerator(_SeedRecorder):
    """``model.generate`` that records the autocast state sampling ran under."""

    def generate(self, input_ids, **kw):
        self.seen = _numerics()
        return super().generate(input_ids, **kw)


def test_sampling_and_the_loss_forward_evaluate_the_policy_under_one_numerics():
    """The samples are on-policy only if the loss scores them with the function that drew them.

    fp32 master weights read in fp32 by the loss and in bf16 by the sampler are
    two functions once a step has moved the weights by less than bf16 resolution.
    """
    sampler = NumericsGenerator()
    trainer.generate(sampler, _PadTokenizer(), ["ab"], _GenArgs())
    scorer = NumericsLM()
    sample = episode_env.Sample(scenario_id="s", prompt="hi", completion="ok", advantage=1.0)
    trainer.sample_loss(scorer, CharTokenizer(), sample, 1, device="cpu")
    assert scorer.seen == [sampler.seen], "the loss forward runs under the sampler's numerics"
    assert sampler.seen == (True, torch.bfloat16)


def test_the_loss_forward_is_compared_with_generation_token_by_token():
    tok, model = CharTokenizer(), TinyLM()
    plain = episode_env.Sample(scenario_id="s", prompt="hi", completion="ok", advantage=1.0)
    prompt_ids, comp_ids = trainer.loss_ids(tok, plain, None)
    with trainer.policy_numerics("cpu"):
        drawn = trainer.completion_logprobs(model, torch.cat([prompt_ids, comp_ids], 1), comp_ids.shape[1])
    exact = trainer.Completion("ok", comp_ids[0].tolist(), prompt_ids=prompt_ids[0].tolist(),
                               logprobs=drawn[0].tolist())
    _, stats = trainer.sample_loss(model, tok, plain.__class__(**{**plain.__dict__, "completion": exact}),
                                   1, device="cpu")
    assert stats["prompt_ids_match"] == 1.0 and stats["gen_max_abs_diff"] < 1e-6
    shifted = list(drawn[0].tolist())
    shifted[1] -= 3.0
    moved = trainer.Completion("ok", comp_ids[0].tolist(), prompt_ids=prompt_ids[0].tolist(), logprobs=shifted)
    _, stats = trainer.sample_loss(model, tok, plain.__class__(**{**plain.__dict__, "completion": moved}),
                                   1, device="cpu")
    assert stats["gen_max_abs_diff"] == pytest.approx(3.0, abs=1e-5) and stats["gen_worst_pos"] == 1.0


def test_a_sample_past_the_generation_gap_is_left_out_of_the_gradient():
    bound = trainer.MAX_GENERATION_GAP
    assert trainer.takes_gradient({"gen_max_abs_diff": bound})
    assert not trainer.takes_gradient({"gen_max_abs_diff": bound + 1e-3})
    assert trainer.takes_gradient({}), "no generation record: nothing to compare, so it trains"
    assert 3.6 > bound, "the gap measured after one Adam step under fp32 scoring is off-policy"


def test_only_samples_that_agree_with_generation_take_a_gradient():
    source = Path(trainer.__file__).read_text()
    body = source[source.index("def _train("):]
    assert body.count("loss.backward()") == 1
    gate = body.index("if takes_gradient(stats):")
    assert 0 < body.index("loss.backward()") - gate < 80, "the backward pass sits under the gate"
    assert 'off_policy, row["loss_vs_generation"]["with_logprobs"])' in body, \
        "the step gate is handed the count of samples left out"


def test_a_systematic_disagreement_with_generation_gates_the_step():
    frac = trainer.MAX_OFF_POLICY_FRACTION
    assert trainer.loss_forward_check(1.0, 10, 3.24, off_policy=1, compared=578)["passed"], \
        "one outlier in 578 samples (measured after the fix) does not refuse the step"
    edge = trainer.loss_forward_check(1.0, 10, 3.0, off_policy=2, compared=100)
    assert edge["passed"] and edge["off_policy_fraction"] == frac
    over = trainer.loss_forward_check(1.0, 10, 3.0, off_policy=3, compared=100)
    assert not over["passed"] and over["off_policy_samples"] == 3
    assert not trainer.loss_forward_check(1.0, 10, 3.6, off_policy=8, compared=43)["passed"], \
        "8 of 43 past 1 nat, as measured after one fp32-scored step, is refused"
    assert trainer.loss_forward_check(1.0, 10, None)["passed"], "no generation record: the NLL bound alone"
    assert not trainer.loss_forward_check(trainer.MAX_SAMPLED_TOKEN_NLL + 1e-3, 10, 0.1, 0, 50)["passed"]


def test_the_default_system_prompt_is_still_the_probe_environments():
    """Narrowness: the probe path is unchanged when no system prompt is passed."""
    tok = RecordingTokenizer()
    trainer.chat_prompt(tok, "x")
    assert tok.systems == [episode_env.SYSTEM]


def test_the_environment_seam_resolves_both_modules_and_refuses_others():
    assert trainer.environment("probe") is episode_env
    assert trainer.environment("rocjitsu") is rj
    with pytest.raises(ValueError):
        trainer.environment("gym")


def test_the_rocjitsu_policy_is_the_turn_budget(scenarios):
    assert trainer.env_policy(rj, 5) == rj.TurnBudget(5)


class _Arg:
    def __init__(self, root, split):
        self.corpus_root, self.split = root, split


def test_the_trainer_loads_the_split_it_is_asked_for(tmp_path):
    root = build_corpus(tmp_path)
    train = trainer.env_scenarios(rj, _Arg(root, None), [])
    test = trainer.env_scenarios(rj, _Arg(root, "test"), [])
    assert {s.kernel for s in train if not s.clean} == {"k_a"}
    assert {s.kernel for s in test if not s.clean} == {"k_b"}


@pytest.mark.parametrize("pool, k", [(190, 16), (10, 3), (7, 7), (5, 0)])
def test_iteration_slices_cover_each_epoch_without_repeats(pool, k):
    items = list(range(pool))
    if k in (0, pool):
        assert trainer.iteration_scenarios(items, k, 1, 0) == items
        return
    chunks = pool // k
    epoch = [trainer.iteration_scenarios(items, k, it, 7) for it in range(1, chunks + 1)]
    seen = [x for chunk in epoch for x in chunk]
    assert all(len(c) == k for c in epoch) and len(seen) == len(set(seen)) == chunks * k
    nxt = trainer.iteration_scenarios(items, k, chunks + 1, 7)
    assert nxt != epoch[0], "a new epoch is a new permutation"


def test_iteration_slices_depend_on_seed_and_iteration_only():
    items = list(range(50))
    a = trainer.iteration_scenarios(items, 8, 4, 123)
    assert a == trainer.iteration_scenarios(list(items), 8, 4, 123)
    assert a != trainer.iteration_scenarios(items, 8, 4, 124)


@pytest.mark.parametrize("flags, fragment", [
    (["--split", "test"], "--split applies to --env rocjitsu"),
    (["--env", "rocjitsu", "--split", "validation"], "--split must be"),
    (["--env", "rocjitsu", "--scenarios-per-iteration", "-1"], "--scenarios-per-iteration"),
])
def test_a_rocjitsu_configuration_that_cannot_mean_what_it_says_is_refused(tmp_path, capsys, flags, fragment):
    code = trainer.main(["--out", str(tmp_path / "run"), *flags])
    assert code == trainer.EXIT_REFUSED
    assert fragment in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_a_rocjitsu_configuration_is_valid():
    args = trainer.build_parser().parse_args(["--out", "x", "--env", "rocjitsu", "--split", "train",
                                              "--scenarios-per-iteration", "16"])
    assert trainer.validate_args(args) is None


def test_a_snapshot_stores_linear_weights_in_bf16_and_everything_else_as_it_is():
    model = TinyLM()
    snap = trainer.snapshot_tensors(model)
    assert snap["head.weight"].dtype == torch.bfloat16
    assert snap["embed.weight"].dtype == torch.float32, "an embedding is read at its own precision"
    assert snap["head.bias"].dtype == torch.float32
    assert torch.equal(snap["head.weight"], model.head.weight.detach().to(torch.bfloat16))
    assert torch.equal(snap["embed.weight"], model.embed.weight.detach())

def test_a_snapshot_loaded_back_gives_the_samplers_forward_exactly():
    model, ids = TinyLM(seed=3), torch.tensor([[3, 9, 1, 40, 7]])
    with trainer.policy_numerics("cpu"):
        live = model(ids).logits
    restored = TinyLM(seed=4)
    restored.load_state_dict({k: v.to(torch.float32) for k, v in trainer.snapshot_tensors(model).items()})
    with trainer.policy_numerics("cpu"):
        again = restored(ids).logits
    assert torch.equal(live, again)

def test_publishing_a_snapshot_leaves_a_complete_tree_and_no_partial(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    class Saves:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, where):
            (Path(where) / self.name).write_text("{}")

    model = TinyLM()
    model.config = Saves("config.json")
    dest = tmp_path / "snapshot-it010"
    trainer.publish_snapshot(model, Saves("tokenizer.json"), dest)
    tensors = load_file(str(dest / "model.safetensors"))
    assert tensors["head.weight"].dtype == torch.bfloat16 and tensors["embed.weight"].dtype == torch.float32
    assert (dest / "config.json").is_file() and (dest / "tokenizer.json").is_file()
    assert not dest.with_name("snapshot-it010.partial").exists()
    trainer.publish_snapshot(model, Saves("tokenizer.json"), dest)
    assert (dest / "model.safetensors").is_file(), "republishing replaces the tree"

def test_a_snapshot_carries_the_policys_generation_config(tmp_path):
    """Without it, transformers samples a snapshot with top_k 50 and no KV cache, unlike the base model."""
    pytest.importorskip("safetensors")

    class Saves:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, where):
            (Path(where) / self.name).write_text(f'{{"from": "{self.name}"}}')

    model = TinyLM()
    model.config = Saves("config.json")
    model.generation_config = Saves("generation_config.json")
    dest = tmp_path / "snapshot-it010"
    trainer.publish_snapshot(model, Saves("tokenizer.json"), dest)
    assert (dest / "generation_config.json").read_text() == '{"from": "generation_config.json"}'

def test_a_negative_snapshot_interval_is_refused():
    args = trainer.build_parser().parse_args(["--out", "x", "--snapshot-every", "-1"])
    assert "snapshot" in (trainer.validate_args(args) or "")

def test_the_sd_floor_keeps_tool_cost_noise_small_and_leaves_real_spread_alone():
    hopeless = [-1.05, -1.10, -1.25, -1.05]  # every episode failed; they differ only in tool costs
    plain, _, _ = trainer.advantages(hopeless)
    floored, _, _ = trainer.advantages(hopeless, sd_floor=1.0)
    assert max(abs(a) for a in plain) > 1.0, "GRPO's normalisation scales tool-cost noise up to unit size"
    assert max(abs(a) for a in floored) < 0.2
    mixed = [5.43, -1.05, -1.10, 2.0]
    assert trainer.advantages(mixed, sd_floor=1.0)[0] == pytest.approx(trainer.advantages(mixed)[0], rel=1e-3)
    assert trainer.advantages([2.0, 2.0], sd_floor=1.0)[0] == [0.0, 0.0]

def test_the_trainer_hands_the_floor_to_the_rollouts():
    source = Path(trainer.__file__).read_text()
    assert 'functools.partial(advantages, sd_floor=getattr(args, "advantage_sd_floor", 0.0))' in source

def test_a_negative_or_unbounded_sd_floor_is_refused():
    for bad in ("-1", "nan", "inf"):
        args = trainer.build_parser().parse_args(["--out", "x", "--advantage-sd-floor", bad])
        assert "sd-floor" in (trainer.validate_args(args) or ""), bad

def test_the_blas_backend_is_set_before_the_model_stack_is_imported(monkeypatch):
    applied = []
    monkeypatch.setattr(trainer.blas_backend, "apply", lambda: applied.append(True))
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError):
        trainer._train(trainer.build_parser().parse_args(["--out", "x"]), None)
    assert applied == [True]
