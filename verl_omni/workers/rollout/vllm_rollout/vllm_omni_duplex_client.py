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
"""WebSocket duplex training client for MiniCPM-o 4.5 Thinker GSPO rollout.

Drives one native-duplex session over the uvicorn server's
``/v1/realtime?duplex=1`` WebSocket endpoint, mirroring
``vllm-omni/examples/online_serving/barge_in_client.py``, and reconstructs the
Thinker policy trajectory (token ids / logprobs / hidden states) plus the
session audio into a verl ``TokenOutput``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import torch
from verl.workers.rollout.replica import TokenOutput

__all__ = ["audio_to_pcm_f32le", "run_duplex_training_session"]


def audio_to_pcm_f32le(audio: Any, sample_rate_hz: int | None = None) -> tuple[bytes, int] | tuple[None, None]:
    """Coerce one dataset audio entry into mono f32-le PCM bytes.

    Accepts ``torch.Tensor``, ``numpy.ndarray`` (float32 in [-1, 1], or int16),
    raw ``bytes``, or a ``soundfile``-loadable path. The caller supplies the
    sample rate; MiniCPM-o 4.5 stage-0 input is 16 kHz mono f32.
    """
    if audio is None:
        return None, None
    sample_rate_hz = int(sample_rate_hz or 16_000)

    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio), sample_rate_hz

    if hasattr(audio, "detach"):
        try:
            audio = audio.detach().cpu().numpy()
        except Exception:  # noqa: BLE001
            pass

    import numpy as np

    try:
        if isinstance(audio, np.ndarray):
            arr = np.asarray(audio)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            arr = arr.reshape(-1)
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            return arr.tobytes(), sample_rate_hz
    except Exception:  # noqa: BLE001
        pass

    if isinstance(audio, str):
        try:
            import soundfile as sf

            arr, file_sr = sf.read(audio, dtype="float32", always_2d=False)
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            arr = np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
            return arr.tobytes(), int(file_sr)
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"cannot load question/ref audio from {audio!r}: {error}") from error

    raise TypeError(f"unsupported duplex audio entry type: {type(audio).__name__}")


def _pcm_f32le_to_pcm16(pcm_f32le: bytes, src_sample_rate_hz: int, dst_sample_rate_hz: int = 16_000) -> bytes:
    """Resample (linear) and quantize mono f32-le PCM to 16 kHz PCM16."""
    import numpy as np

    samples = np.frombuffer(pcm_f32le, dtype=np.float32)
    if src_sample_rate_hz != dst_sample_rate_hz and samples.size > 1:
        target_len = round(samples.size * dst_sample_rate_hz / src_sample_rate_hz)
        positions = np.linspace(0.0, samples.size - 1, num=target_len)
        samples = np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _pcm16_to_wav_bytes(pcm16: bytes, sample_rate_hz: int = 16_000) -> bytes:
    """Wrap mono PCM16 bytes in a WAV container (for ``ref_audio`` data URLs)."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm16)
    return buffer.getvalue()


def _deserialize_hidden_states(payload: dict[str, Any]) -> torch.Tensor | None:
    """Rebuild a Thinker hidden tensor from the WebSocket wire payload."""
    import base64

    import numpy as np

    shape = payload.get("shape")
    data = payload.get("data")
    if not isinstance(shape, list) or not isinstance(data, str) or not data:
        return None
    if any(not isinstance(dim, (int, float)) or int(dim) <= 0 for dim in shape):
        return None
    raw = base64.b64decode(data)
    dtype = str(payload.get("dtype") or "")
    if dtype == "torch.bfloat16":
        tensor = torch.from_numpy(np.frombuffer(raw, dtype=np.int16)).view(torch.bfloat16)
    elif dtype == "torch.float16":
        tensor = torch.from_numpy(np.frombuffer(raw, dtype=np.int16)).view(torch.float16)
    elif dtype == "torch.float32":
        tensor = torch.from_numpy(np.frombuffer(raw, dtype=np.int32)).view(torch.float32)
    else:
        tensor = torch.from_numpy(np.frombuffer(raw, dtype=np.float32))
    return tensor.reshape(tuple(int(dim) for dim in shape))


def _map_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
    """Map a vLLM finish reason to verl's stop-reason vocabulary."""
    if finish_reason == "abort":
        return "aborted"
    if finish_reason in ("stop", "length"):
        return "completed"
    return finish_reason


async def run_duplex_training_session(
    *,
    server_address: str,
    server_port: int,
    model: str,
    request_id: str,
    audio_data: Optional[list[Any]],
    response_length: Optional[int] = None,
    timeout: float = 30.0,
) -> TokenOutput:
    """Run one native-duplex session over the server's WebSocket endpoint.

    Mirrors ``vllm-omni/examples/online_serving/barge_in_client.py``: a
    ``DuplexClient`` connects to ``/v1/realtime?duplex=1`` and drives one
    session (open -> stream question -> commit -> collect -> close). Thinker
    token ids / logprobs / hidden states ride the ``response.output_audio.delta``
    events' ``metadata.vllm_omni`` payload projected by the vLLM-Omni serving
    runtime bridge.

    ``audio_data`` carries ``[question_audio, ref_audio]``: the first entry is
    the user utterance to stream, the optional second is the reference voice.
    """
    from vllm_omni.clients.duplex import (
        DuplexClient,
        EventCollector,
        audio_data_url,
        wait_for_condition,
    )
    from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

    audios = list(audio_data) if audio_data else []
    input_sample_rate = 16_000
    question_pcm, question_sr = audio_to_pcm_f32le(audios[0] if audios else None, input_sample_rate)
    ref_pcm, ref_sr = audio_to_pcm_f32le(audios[1] if len(audios) > 1 else None, input_sample_rate)
    if question_pcm is None:
        raise ValueError("duplex generate requires a question audio as the first audio entry.")

    question_sr = int(question_sr or input_sample_rate)
    ref_sr = int(ref_sr or input_sample_rate)
    question_pcm16 = _pcm_f32le_to_pcm16(question_pcm, question_sr, input_sample_rate)
    ref_data_url = None
    if ref_pcm:
        ref_pcm16 = _pcm_f32le_to_pcm16(ref_pcm, ref_sr, input_sample_rate)
        ref_data_url = audio_data_url(_pcm16_to_wav_bytes(ref_pcm16, input_sample_rate))

    if ":" in server_address and not server_address.startswith("["):
        server_address = f"[{server_address}]"
    url = f"ws://{server_address}:{server_port}/v1/realtime"

    session_config = create_duplex_session_config(ref_audio=ref_data_url)
    collector = EventCollector()
    client = DuplexClient(
        url,
        model=model,
        config=session_config,
        session_id=request_id,
        reconnect=None,
        heartbeat_interval_s=None,
        handshake_timeout_s=timeout,
    )
    async with client:
        consume_task = asyncio.create_task(collector.consume(client))
        await client.stream_pcm(question_pcm16, chunk_ms=100, realtime=False)
        await client.commit(create_response=False)
        await wait_for_condition(
            lambda: collector.count("response.done") > 0,
            timeout_s=timeout,
            label="response.done",
        )
        await client.close(timeout_s=timeout)
        try:
            await asyncio.wait_for(consume_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            consume_task.cancel()

    # Reconstruct the Thinker policy trajectory from the collected events.
    token_ids: list[int] = []
    log_probs: list[float] = []
    hidden_chunks: list[torch.Tensor] = []
    finish_reason: str | None = None
    for event in collector.events:
        event_type = event.get("type")
        if event_type == "response.output_audio.delta":
            vllm_omni = (event.get("metadata") or {}).get("vllm_omni") or {}
            ids = vllm_omni.get("thinker_token_ids") or []
            logps = vllm_omni.get("thinker_logprobs") or []
            hidden = vllm_omni.get("thinker_hidden_states")
            if ids:
                token_ids.extend(int(value) for value in ids)
            if logps:
                log_probs.extend(float(value) for value in logps)
            if isinstance(hidden, dict):
                tensor = _deserialize_hidden_states(hidden)
                if tensor is not None:
                    hidden_chunks.append(tensor)
        elif event_type == "response.done" and finish_reason is None:
            status = (event.get("response") or {}).get("status")
            finish_reason = "abort" if status == "cancelled" else "stop"

    hidden_states = torch.cat(hidden_chunks, dim=0) if hidden_chunks else None
    if response_length is not None:
        token_ids = token_ids[:response_length]
        if log_probs:
            log_probs = log_probs[:response_length]
    if not token_ids:
        raise RuntimeError("MiniCPM-o 4.5 duplex generate produced an empty Thinker policy trajectory.")

    extra_fields = {
        "audio": collector.audio_bytes(),
        "audio_sample_rate": collector.output_sample_rate_hz,
        "transcript": "".join(collector.response_text(rid) for rid in collector.response_ids) or None,
        "finish_reason": finish_reason,
    }
    if hidden_states is not None:
        extra_fields["thinker_hidden_states"] = hidden_states

    return TokenOutput(
        token_ids=token_ids,
        log_probs=log_probs or None,
        stop_reason=_map_stop_reason(finish_reason),
        num_preempted=None,
        extra_fields=extra_fields,
    )
