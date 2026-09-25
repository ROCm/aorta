"""Prompt profiles: which messages a real proposer sends the model.

The probe loop's state -- the cell summaries, the remaining candidates and the
names already tried -- is fixed by the loop. A *profile* decides how that state
is written down for the model: the system prompt, the layout of the user
message, and the handful of request fields that change what the model is
conditioned on. Parsing and validating the reply do not vary with the profile;
every profile's reply goes through the same ``_step_from_content`` and
``AgentPolicy.validate_step``.

``default``
    The prompt ``aorta agent`` has always sent (``llm._build_prompt``). It is
    the right choice for any general-purpose model and the only profile a
    user who has not trained a model should use.

``rl-episode``
    The prompt the probe policy is post-trained on: the episode environment's
    system prompt and user message, byte for byte, with thinking disabled the
    way training generated. A checkpoint trained on it answers it much better
    than it answers ``default``, because the gain from post-training is tied to
    the words the policy learned on. A general model does *worse* on it, so it
    is opt-in and never the default.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

DEFAULT_PROMPT_PROFILE = "default"
RL_EPISODE_PROMPT_PROFILE = "rl-episode"

#: The system prompt of the probe policy's episode environment, frozen as it
#: was trained: ``SYSTEM`` in the multi-step episode environment
#: (``examples/rl/episode_env.py``, #525), sent at every step of an episode.
#:
#: This is data, not prose. A checkpoint post-trained on this text has learned
#: to answer *this* string, and the gain does not carry over to other wording:
#: the same checkpoint that improves markedly here does no better than its
#: base model on ``default``. Rewording it, re-wrapping it, or regenerating the
#: category list from ``PROBE_CATEGORIES`` silently detaches every checkpoint
#: trained on it from the prompt it learned, and nothing fails -- the replies
#: still parse, they are just worse. ``RL_EPISODE_SYSTEM_SHA256`` and its test
#: exist so that an edit fails loudly instead. Change it only together with a
#: checkpoint trained on the new text, and update the digest in the same commit.
#:
#: The lists are written out rather than derived for the same reason. The
#: verdicts are the probe classifier's; the categories are the eight
#: ``PROBE_CATEGORIES`` names as the trainer rendered them, which
#: ``tests/agent/test_prompt_profiles.py`` checks still validate.
#:
#: It asks for a ``verdict`` and a ``detectors`` list the loop does not read.
#: They are part of what the policy was trained to produce, so they stay in the
#: prompt; ``AgentStep.from_dict`` ignores keys it does not know.
RL_EPISODE_SYSTEM = (
    "You are an AORTA probe agent. You are shown the evidence from one probe "
    "cell and a list of registered mitigation names.\n"
    "Do two things in one reply.\n"
    "(a) Triage the evidence: say what happened and cite the finding IDs that "
    "justify it. `verdict` must be one of ['pass', 'warn', 'fail', 'error', "
    "'not_checked']. `detectors` must be a list of the finding IDs you were "
    "shown that fired, written exactly as they appear, and nothing else; a "
    "clean run cites an empty list.\n"
    "(b) Propose what to try next: `next_mitigations` must contain ONLY names "
    "from the candidate list, and `category` must be one of "
    "['checkpoint_race', 'illegal_mem', 'launch_error', 'oom_fragment', "
    "'perf_regression', 'rccl_hang', 'thermal_throttle', 'unknown'].\n"
    "Return strict JSON with exactly these keys: verdict (string), detectors "
    "(list of strings), category (string), hypothesis (string), "
    "next_mitigations (list of strings), confidence (number 0-1), stop "
    "(boolean)."
)

#: sha256 of ``RL_EPISODE_SYSTEM`` as trained, UTF-8.
RL_EPISODE_SYSTEM_SHA256 = "72f18e5541a8a6bbc7ef3653a6d9b7243ba41422610f7aa1766fc6e7f83421c6"


def _rl_episode_cell_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    """One loop cell summary, in the shape the policy was trained on.

    Not the loop's shape. Training unions the failure and warn detector lists
    into one sorted ``failure_detectors_fired`` and sends no
    ``warn_detectors_fired``; the loop keeps two lists in first-seen order.
    What the model loses is the failure/warn split, which it never saw.
    """
    detectors = set(row.get("failure_detectors_fired") or [])
    detectors.update(row.get("warn_detectors_fired") or [])
    return {
        "cell_name": row.get("cell_name"),
        "verdict": row.get("verdict"),
        "exit_code": row.get("exit_code"),
        "failure_detectors_fired": sorted(str(d) for d in detectors),
        "capture": row.get("capture") or {},
    }


def build_rl_episode_prompt(
    cell_summaries: list[dict[str, Any]], remaining: list[str]
) -> tuple[str, str]:
    """The ``rl-episode`` system and user messages for one loop state.

    Two keys and a ``sort_keys`` dump, as trained. ``symptom`` and
    ``already_tried`` are not sent: the policy never saw a symptom, and what
    was tried is on screen as the cells that ran. ``remaining`` is the loop's
    own remaining-candidate list, so a tried name drops out of it exactly as
    it did in training.
    """
    user = json.dumps(
        {
            "cell_summaries": [_rl_episode_cell_summary(row) for row in cell_summaries],
            "candidates": list(remaining),
        },
        sort_keys=True,
    )
    return RL_EPISODE_SYSTEM, user


@dataclass(frozen=True)
class PromptProfile:
    """How a real proposer writes the loop's state down for the model."""

    name: str
    #: Builds ``(system, user)`` from the cell summaries and the remaining
    #: candidates. ``None`` is the shipped ``llm._build_prompt``.
    build: Callable[[list[dict[str, Any]], list[str]], tuple[str, str]] | None
    #: Ask the chat template not to open a reasoning block.
    disable_thinking: bool
    #: Whether the direct LiteLLM path constrains the reply with
    #: ``response_format={"type": "json_object"}``.
    json_mode: bool

    def extra_body(self) -> dict[str, Any]:
        """Fields to merge into the chat-completions request body.

        A fresh dict per call, since the HTTP client may keep what it is given.
        ``chat_template_kwargs`` is what vLLM and TokenSpeed read; a server
        that does not know it ignores it or rejects the request, which is why
        only ``rl-episode`` -- meant for a model you serve yourself -- sends it.
        """
        if self.disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}


#: Every profile, by the name ``--prompt-profile`` takes.
#:
#: ``rl-episode`` disables thinking and drops JSON mode because that is how the
#: policy generated in training: no reasoning block, no grammar. JSON mode with
#: thinking disabled is also the one request shape that a server-side Qwen3
#: reasoning parser mis-handles on TokenSpeed (3-4% of replies parse), and the
#: parser is the recommended way to serve a Qwen3 model for ``default``.
PROMPT_PROFILES: Mapping[str, PromptProfile] = MappingProxyType(
    {
        DEFAULT_PROMPT_PROFILE: PromptProfile(
            name=DEFAULT_PROMPT_PROFILE,
            build=None,
            disable_thinking=False,
            json_mode=True,
        ),
        RL_EPISODE_PROMPT_PROFILE: PromptProfile(
            name=RL_EPISODE_PROMPT_PROFILE,
            build=build_rl_episode_prompt,
            disable_thinking=True,
            json_mode=False,
        ),
    }
)


def get_prompt_profile(name: str) -> PromptProfile:
    """The profile called ``name``, or ``ValueError`` naming the valid ones.

    Fails closed: a misspelt profile must not fall back to ``default``, because
    the symptom of the wrong prompt is a model that quietly answers worse.
    """
    try:
        return PROMPT_PROFILES[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown agent prompt profile: {name!r} "
            f"(expected one of {', '.join(sorted(PROMPT_PROFILES))})"
        ) from None


__all__ = [
    "DEFAULT_PROMPT_PROFILE",
    "PROMPT_PROFILES",
    "RL_EPISODE_PROMPT_PROFILE",
    "RL_EPISODE_SYSTEM",
    "RL_EPISODE_SYSTEM_SHA256",
    "PromptProfile",
    "build_rl_episode_prompt",
    "get_prompt_profile",
]
