"""The trainer's CPU-testable half: the loss, the guard, the checks, the refusals.

The training loop itself needs a model and a GPU. Everything it *decides* does
not, so it is factored out and tested here against a tiny causal LM built in
the test and a character tokenizer -- no ``transformers``, no download, no
device. Skipped where torch is absent; CI's CPU lane installs a CPU wheel.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "examples"))

import episode_env  # noqa: E402
import rescore_episodes  # noqa: E402
import train_grpo_step as trainer  # noqa: E402
from episode_env import Sample  # noqa: E402
from test_episode_env import MENU, build_archive, make_scenario, reply  # noqa: E402

from aorta.agent.policy import AgentPolicy  # noqa: E402


class CharTokenizer:
    """Enough of a HF tokenizer for ``chat_prompt`` and ``sample_loss``."""

    eos_token_id = 0
    pad_token_id = 0

    def __init__(self) -> None:
        self.template_kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        ids = [1 + (ord(c) % 60) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class TinyLM(torch.nn.Module):
    """An embedding and a head: a causal LM in the only sense ``sample_loss`` needs."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(64, 16)
        self.head = torch.nn.Linear(16, 64)

    def forward(self, input_ids):
        class Out:
            pass

        out = Out()
        out.logits = self.head(self.embed(input_ids))
        return out


def sample(advantage: float, completion: str = '{"x": 1}') -> Sample:
    return Sample(scenario_id="s", prompt="p", completion=completion, advantage=advantage)


# ---------------------------------------------------------------------------
# advantages and the prompt
# ---------------------------------------------------------------------------


def test_the_normaliser_agrees_with_the_cpu_copy_the_reports_use():
    for values in ([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [4.0], [-3.0, 7.0]):
        assert trainer.advantages(values) == pytest.approx(rescore_episodes.advantages(values))


def test_a_flat_group_has_zero_advantage_not_a_division_blow_up():
    adv, _, sd = trainer.advantages([2.0, 2.0, 2.0])
    assert sd == 0.0 and adv == [0.0, 0.0, 0.0]


def test_thinking_is_disabled_and_the_system_prompt_is_the_environments():
    tok = CharTokenizer()
    text = trainer.chat_prompt(tok, "hello")
    assert tok.template_kwargs["enable_thinking"] is False
    assert episode_env.SYSTEM in text and "hello" in text


# ---------------------------------------------------------------------------
# FiniteLogits
# ---------------------------------------------------------------------------


def test_a_nan_becomes_a_token_that_is_never_drawn():
    guard = trainer.FiniteLogits(eos_token_id=0)
    scores = torch.tensor([[1.0, float("nan"), 2.0]])
    out = guard(None, scores)
    assert out[0, 1] == float("-inf") and out[0, 2] == 2.0
    assert (guard.nan_steps, guard.nan_rows, guard.dead_rows) == (1, 1, 0)


def test_positive_infinity_is_damage_too():
    """softmax over a row holding +inf is inf/inf = NaN, so +inf beside finite
    logits crashes multinomial exactly as a NaN does."""
    guard = trainer.FiniteLogits(eos_token_id=0)
    out = guard(None, torch.tensor([[float("inf"), 0.0, 1.0]]))
    assert out[0].tolist() == [float("-inf"), 0.0, 1.0]
    assert bool(torch.isfinite(torch.softmax(out, dim=-1)).all())
    assert guard.nan_steps == 1


def test_top_p_minus_infinity_is_expected_and_left_alone():
    """Narrowness: the guard runs after top-p, whose -inf entries are output."""
    guard = trainer.FiniteLogits(eos_token_id=0)
    scores = torch.tensor([[float("-inf"), 0.5, float("-inf")]])
    assert torch.equal(guard(None, scores), scores)
    assert (guard.nan_steps, guard.dead_rows) == (0, 0)


def test_a_row_with_nothing_finite_is_ended_at_eos_not_flattened():
    guard = trainer.FiniteLogits(eos_token_id=2)
    scores = torch.tensor([[float("nan"), float("-inf"), float("nan")], [0.0, 1.0, 2.0]])
    out = guard(None, scores)
    assert out[0].tolist() == [float("-inf"), float("-inf"), 0.0]
    assert out[1].tolist() == [0.0, 1.0, 2.0]
    assert guard.dead_rows == 1
    # And the repaired row is one `multinomial` accepts, and it draws EOS.
    assert torch.multinomial(torch.softmax(out[:1], dim=-1), 1).item() == 2


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------


def test_a_zero_advantage_or_empty_completion_contributes_nothing():
    model, tok = TinyLM(), CharTokenizer()
    assert trainer.sample_loss(model, tok, sample(0.0), 4, device="cpu") is None
    assert trainer.sample_loss(model, tok, sample(1.0, ""), 4, device="cpu") is None


def test_a_zero_advantage_still_carries_the_kl_term():
    """KL does not depend on the advantage: a flat group must still be anchored
    to the reference, and its loss is the KL term alone."""
    tok = CharTokenizer()
    result = trainer.sample_loss(TinyLM(3), tok, sample(0.0), 2, device="cpu",
                                 reference=TinyLM(4), kl_beta=0.5)
    assert result is not None
    loss, stats = result
    assert stats["kl_sum"] > 0 and stats["pg_abs"] == 0.0
    assert float(loss) == pytest.approx(0.5 / 2 * stats["kl_sum"], rel=1e-5)


def test_the_gradient_raises_the_completion_likelihood_for_a_positive_advantage():
    """One SGD step on ``-(A/N) * log pi`` must move log pi in the sign of A."""
    tok = CharTokenizer()
    for advantage in (1.0, -1.0):
        model = TinyLM(seed=1)
        before = -trainer.sample_loss(model, tok, sample(1.0), 1, device="cpu")[1]["nll"]
        loss, _ = trainer.sample_loss(model, tok, sample(advantage), 1, device="cpu")
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= 0.5 * p.grad
        after = -trainer.sample_loss(model, tok, sample(1.0), 1, device="cpu")[1]["nll"]
        assert (after - before) * advantage > 0, advantage


def test_the_sampled_ids_run_through_the_first_stop_token_and_no_further():
    assert trainer.sampled_ids([5, 6, 0, 0, 0], {0}) == [5, 6, 0], "padding is not output"
    assert trainer.sampled_ids([5, 6, 7], {0}) == [5, 6, 7], "hit the length cap"
    assert trainer.sampled_ids([5, 9, 6, 0], {0, 9}) == [5, 9], "either stop token ends it"


def test_the_stop_set_includes_every_configured_eos():
    class Config:
        eos_token_id = [11, 12]

    class Model:
        generation_config = Config()

    assert trainer._stop_ids(Model(), CharTokenizer()) == {0, 11, 12}


def test_a_completion_is_its_text_and_carries_its_ids():
    reply = trainer.Completion('{"x": 1}', [3, 4])
    assert reply == '{"x": 1}' and isinstance(reply, str)
    assert reply.token_ids == (3, 4)
    assert json.loads(json.dumps({"raw": reply}))["raw"] == '{"x": 1}'


def test_the_loss_is_taken_over_the_sampled_ids_not_a_re_tokenisation():
    """Decode-then-encode need not return the sampled IDs; the gradient has to
    be over what was drawn."""
    model, tok = TinyLM(seed=5), CharTokenizer()
    text = '{"x": 1}'
    retokenised = trainer.sample_loss(model, tok, sample(1.0, text), 1, device="cpu")[1]
    drawn = [7, 8, 9]
    sampled = Sample(scenario_id="s", prompt="p", completion=trainer.Completion(text, drawn),
                     advantage=1.0)
    _, stats = trainer.sample_loss(model, tok, sampled, 1, device="cpu")
    prompt_ids = tok(trainer.chat_prompt(tok, "p"))["input_ids"]
    ids = torch.cat([prompt_ids, torch.tensor([drawn])], dim=1)
    logp = torch.log_softmax(model(ids).logits[:, :-1, :], dim=-1)
    want = -logp.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, prompt_ids.shape[1] - 1:].sum()
    assert stats["nll"] == pytest.approx(float(want), rel=1e-5)
    assert stats["nll"] != pytest.approx(retokenised["nll"])


def test_the_environment_hands_the_completion_object_through_to_the_sample(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=MENU[1]))
    _group, samples, wire = episode_env.rollout_scenario(
        scenario, 2, AgentPolicy(max_iterations=8),
        lambda users: [trainer.Completion(reply([MENU[1]]), [1, 2])] * len(users),
        advantage_fn=trainer.advantages,
    )
    assert samples and all(s.completion.token_ids == (1, 2) for s in samples)
    assert json.dumps(wire)


def test_the_loss_scales_with_one_over_the_sample_count():
    model, tok = TinyLM(), CharTokenizer()
    one, _ = trainer.sample_loss(model, tok, sample(1.0), 1, device="cpu")
    four, _ = trainer.sample_loss(model, tok, sample(1.0), 4, device="cpu")
    assert float(one) == pytest.approx(4 * float(four))


def test_kl_to_an_identical_reference_is_zero_and_to_another_is_positive():
    tok = CharTokenizer()
    policy = TinyLM(seed=3)
    same = TinyLM(seed=3)
    other = TinyLM(seed=4)
    _, stats = trainer.sample_loss(policy, tok, sample(1.0), 1, device="cpu",
                                   reference=same, kl_beta=0.1)
    assert stats["kl_sum"] == pytest.approx(0.0, abs=1e-6)
    assert stats["kl_tokens"] > 0
    _, stats = trainer.sample_loss(policy, tok, sample(1.0), 1, device="cpu",
                                   reference=other, kl_beta=0.1)
    assert stats["kl_sum"] > 0


def test_the_kl_term_is_off_at_beta_zero_even_with_a_reference():
    tok = CharTokenizer()
    _, stats = trainer.sample_loss(TinyLM(3), tok, sample(1.0), 1, device="cpu",
                                   reference=TinyLM(4), kl_beta=0.0)
    assert stats["kl_tokens"] == 0.0


def test_the_k3_estimator_is_non_negative_per_token():
    """So the penalty cannot pay the policy for drifting."""
    r = torch.linspace(-5, 5, 101)
    assert bool(((torch.exp(r) - r - 1.0) >= 0).all())


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def fp(*values):
    return {f"t{i}": (v, v, v) for i, v in enumerate(values)}


def checks(pre, post, frozen_pre=None, frozen_post=None, grad_norm=1.0, cosine=0.5):
    frozen_pre = {"emb": (1.0, 1.0, 1.0)} if frozen_pre is None else frozen_pre
    frozen_post = dict(frozen_pre) if frozen_post is None else frozen_post
    return trainer.update_checks(pre, post, frozen_pre, frozen_post, grad_norm, cosine)


def gating_passed(result):
    return all(result[k]["passed"] for k in trainer.GATING_CHECKS)


def test_a_clean_update_passes_every_gating_check():
    assert gating_passed(checks(fp(1.0, 2.0), fp(1.1, 2.1)))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_weight_after_the_step_fails_even_though_it_moved(bad):
    """A NaN fingerprint compares unequal to everything, so "moved" passes on
    exactly the tensor that is damaged; the finiteness check is what catches it."""
    result = checks(fp(1.0, 2.0), fp(1.1, bad))
    assert result["every_trained_tensor_moved"]["passed"], "the defect: it reads as moved"
    assert not result["trained_tensors_are_finite"]["passed"]
    assert result["trained_tensors_are_finite"]["non_finite_examples"] == ["t1"]
    assert not gating_passed(result)
    assert "trained_tensors_are_finite" in trainer.GATING_CHECKS


def test_large_finite_weights_are_finite():
    """Narrowness: magnitude is not damage."""
    assert checks(fp(1.0), fp(1e30))["trained_tensors_are_finite"]["passed"]


def test_an_unmoved_trained_tensor_fails():
    result = checks(fp(1.0, 2.0), fp(1.1, 2.0))
    assert not result["every_trained_tensor_moved"]["passed"]
    assert result["every_trained_tensor_moved"]["unmoved_examples"] == ["t1"]


def test_a_moved_frozen_control_fails():
    result = checks(fp(1.0), fp(1.1), frozen_post={"emb": (1.0, 1.0, 1.5)})
    assert not result["frozen_control_unchanged"]["passed"]


def test_no_frozen_control_is_a_failure_not_a_vacuous_pass():
    assert not checks(fp(1.0), fp(1.1), frozen_pre={}, frozen_post={})[
        "frozen_control_unchanged"]["passed"]


@pytest.mark.parametrize("norm", [0.0, math.nan, math.inf])
def test_a_zero_or_non_finite_gradient_fails(norm):
    assert not checks(fp(1.0), fp(1.1), grad_norm=norm)["gradient_is_finite_and_nonzero"]["passed"]


def test_a_negative_cosine_is_advisory_and_does_not_gate():
    result = checks(fp(1.0), fp(1.1), cosine=-0.02)
    assert not result["step_descends_the_gradient"]["passed"]
    assert result["step_descends_the_gradient"]["advisory"] is True
    assert gating_passed(result)
    assert "step_descends_the_gradient" not in trainer.GATING_CHECKS


def test_the_fingerprint_sees_a_change_that_preserves_the_sum():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([0.0, 3.0, 3.0])
    assert float(a.sum()) == float(b.sum())
    assert trainer.fingerprint(a) != trainer.fingerprint(b)


# ---------------------------------------------------------------------------
# the rollout wrapper and the refusals
# ---------------------------------------------------------------------------


def test_episode_rollouts_templates_each_prompt_and_delegates_to_the_env(tmp_path, monkeypatch):
    scenario = make_scenario(build_archive(tmp_path, resolver=MENU[2]))
    seen: list[list[str]] = []

    def fake_generate(model, tok, prompts, args):
        seen.append(prompts)
        return [reply([MENU[len(seen)]])] * len(prompts)

    monkeypatch.setattr(trainer, "generate", fake_generate)

    class Args:
        group = 3
        log_episodes = 0

    groups, samples, wire = trainer.episode_rollouts(
        None, CharTokenizer(), [scenario], Args(), AgentPolicy(max_iterations=8)
    )
    assert groups[0]["converged_rate"] == 1.0
    assert len(samples) == len(wire) == 3 * 2
    assert all(p.startswith("<system>") for batch in seen for p in batch)


@pytest.mark.parametrize("flags, fragment", [
    (["--iterations", "0"], "--iterations"),
    (["--group", "1"], "--group"),
    (["--lr", "0"], "--lr"),
    (["--kl-beta", "-0.1"], "--kl-beta"),
    (["--top-p", "1.5"], "--top-p"),
    (["--temperature", "0"], "--temperature"),
    (["--clip", "-1"], "--clip"),
    (["--clip", "nan"], "--clip"),
    (["--kl-beta", "nan"], "--kl-beta"),
])
def test_a_configuration_that_cannot_produce_a_checked_update_is_refused(
    tmp_path, capsys, flags, fragment
):
    code = trainer.main(["--out", str(tmp_path / "run"), *flags])
    assert code == trainer.EXIT_REFUSED
    assert fragment in capsys.readouterr().err
    assert not (tmp_path / "run").exists(), "refused before touching disk"


@pytest.mark.parametrize("artifact", trainer.RUN_ARTIFACTS)
def test_an_out_dir_holding_another_runs_artifact_is_refused(tmp_path, artifact):
    (tmp_path / artifact).mkdir()
    assert artifact in trainer.check_output_dir(tmp_path)


def test_an_empty_or_unrelated_out_dir_is_accepted(tmp_path):
    assert trainer.check_output_dir(tmp_path / "new") is None
    (tmp_path / "notes.txt").write_text("x")
    assert trainer.check_output_dir(tmp_path) is None


def test_a_fresh_run_into_a_used_out_is_refused_before_it_touches_anything(tmp_path, capsys):
    out = tmp_path / "run"
    out.mkdir()
    (out / "wire.jsonl").write_text('{"iteration": 1}\n')
    assert trainer.main(["--out", str(out)]) == trainer.EXIT_REFUSED
    assert "already holds ['wire.jsonl']" in capsys.readouterr().err
    assert (out / "wire.jsonl").read_text() == '{"iteration": 1}\n'


def test_a_positive_clip_is_a_valid_configuration():
    """Narrowness: the refusal is for negative and non-finite clips only."""
    args = trainer.build_parser().parse_args(["--out", "x", "--clip", "1.0"])
    assert trainer.validate_args(args) is None


def test_the_shipped_defaults_are_a_valid_configuration():
    """Narrowness for the refusals above."""
    assert trainer.validate_args(trainer.build_parser().parse_args(["--out", "x"])) is None


def test_the_exit_codes_are_distinct_and_none_is_argparses():
    assert len({0, trainer.EXIT_FAILED, trainer.EXIT_REFUSED}) == 3
    assert 2 not in {trainer.EXIT_FAILED, trainer.EXIT_REFUSED}
