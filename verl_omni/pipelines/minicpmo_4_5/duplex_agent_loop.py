# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WebSocket duplex agent loop for MiniCPM-o 4.5 Thinker GSPO (Path 2).

The rollout drives the in-process vLLM-Omni engine through the rollout
server's ``/v1/realtime`` WebSocket endpoint instead of the batched
``generate`` path. This loop only flags the request as duplex
(``sampling_params["duplex"] = True``) and maps the resulting
``TokenOutput`` into an ``AgentLoopOutput``; the full session flow (open ->
stream question -> commit -> collect -> close) and the Thinker trajectory
reconstruction live in
:func:`verl_omni.workers.rollout.vllm_rollout.vllm_omni_duplex_client.run_duplex_training_session`,
executed inside the rollout server actor.

The Thinker (stage 0) text/turn token sequence is the policy; per-token
logprobs come from the Thinker native-duplex sampler. The Talker (stage 1)
and Code2Wav (stage 2) stay frozen and only the stage-2 waveform is
collected, so the session-level reward can score the audio.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("minicpmo_duplex_agent")
class MiniCPMODuplexAgentLoop(AgentLoopBase):
    """Native-duplex single-turn agent loop for MiniCPM-o 4.5 Thinker GSPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios") or []
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # Build the actor-side prompt (placeholder ids + multimodal inputs) so
        # the FSDP training forward can recompute Thinker logprobs on the same
        # vocabulary. The duplex rollout itself consumes the raw audio on the
        # server side; ``run_duplex_training_session`` requires the question
        # audio as the first ``audio_data`` entry (optional reference voice
        # second) and raises when it is missing.
        self._assert_mm_supported(bool(multi_modal_data))
        prompt_ids = await self.ct_build_initial_tokens(
            messages,
            images=images,
            videos=videos,
            audios=audios,
        )

        # Flag the request so the rollout server drives one native-duplex
        # session over its own /v1/realtime WebSocket endpoint
        # (vLLMOmniHttpServer.generate -> _generate_duplex). The server
        # manager owns the acquire/release cycle around that call.
        request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
        duplex_sampling_params = dict(sampling_params)
        duplex_sampling_params["duplex"] = True

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=duplex_sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        token_ids = list(output.token_ids)
        if not token_ids:
            raise RuntimeError("MiniCPM-o 4.5 duplex session produced an empty Thinker policy trajectory.")

        response_ids = token_ids[: self.response_length]
        response_logprobs = list(output.log_probs)[: self.response_length] if output.log_probs is not None else None
        response_mask = [1] * len(response_ids)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=dict(output.extra_fields),
        )
