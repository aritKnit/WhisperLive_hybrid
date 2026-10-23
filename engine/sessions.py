"""Sessions that drive model-neutral backends.

``WindowedSession``  - offline backends: re-transcribe a growing window, commit on a
                       VAD endpoint or when the window hits a cap.
``StreamingSession`` - native streaming backends: push chunks straight in.

Both reuse ``ServeClientBase`` for transport (websocket send, segment formatting,
translation queue, finalizer, cleanup). The legacy Whisper sessions keep using the
loop that still lives in ``ServeClientBase``.
"""
import json
import logging
import queue
import threading
import time

import numpy as np

from whisper_live import metrics as wl_metrics
from whisper_live.backend.base import ServeClientBase
from whisper_live.engine.base import OfflineBackend, StreamingBackend
from whisper_live.engine.vad_stage import SessionVAD

FLUSH_TIMEOUT_S = 5.0


class _EngineSession(ServeClientBase):
    """Shared plumbing for the two session kinds."""

    def __init__(self, websocket, backend_name, capabilities, client_uid, language=None,
                 task="transcribe", use_vad=True, vad=None, vad_threshold=0.5,
                 min_endpoint_silence_s=0.6, send_last_n_segments=10, no_speech_thresh=0.45,
                 clip_audio=False, same_output_threshold=10, translation_queue=None,
                 diarization=None, word_timestamps=False):
        super().__init__(client_uid, websocket, send_last_n_segments, no_speech_thresh,
                         clip_audio, same_output_threshold, translation_queue,
                         diarization, word_timestamps)
        self.backend_name = backend_name
        self.capabilities = capabilities
        self.language = language
        self.task = task
        self.use_vad = bool(use_vad)
        self.min_endpoint_silence_s = min_endpoint_silence_s
        self.vad = (vad or SessionVAD(sample_rate=self.RATE, threshold=vad_threshold)) if self.use_vad else None
        self._announced_language = None
        self._flush = threading.Event()
        self._flushed = threading.Event()

    # ---- handshake -------------------------------------------------------------
    def _start(self):
        self.trans_thread = threading.Thread(target=self.speech_to_text, daemon=True)
        self.trans_thread.start()
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": self.backend_name,
            "capabilities": self.capabilities.to_dict(),
        }))

    # ---- helpers ---------------------------------------------------------------
    def _announce_language(self, segments):
        for s in segments:
            lang = getattr(s, "language", None)
            if lang and lang != self._announced_language:
                self._announced_language = lang
                try:
                    self.websocket.send(json.dumps({
                        "uid": self.client_uid, "language": lang,
                        "language_prob": getattr(s, "language_probability", None),
                    }))
                except Exception as e:  # client gone; the main loop notices separately
                    logging.info(f"Could not send language message: {e}")
                return

    def _commit(self, segment, base, duration):
        """Append one finished segment to the transcript and the translation queue."""
        start = base + max(0.0, segment.start)
        end = base + (min(duration, segment.end) if segment.end > segment.start else duration)
        if end <= start:
            end = base + duration
        words = self._extract_words(segment, base)
        done = self.format_segment(start, end, segment.text.strip(), completed=True, words=words)
        self.transcript.append(done)
        self.text.append(segment.text)
        if self.translation_queue:
            try:
                self.translation_queue.put(done.copy(), timeout=0.1)
            except queue.Full:
                logging.warning("Translation queue is full, skipping segment")
        return done

    def end_of_audio(self):
        """Called when the client says END_OF_AUDIO: commit whatever is still pending."""
        if self.exit or not self._has_audio():
            return  # nothing was ever sent: do not hold the seat waiting for a flush
        self._flushed.clear()
        self._flush.set()
        self.frames_ready.set()
        self._flushed.wait(timeout=FLUSH_TIMEOUT_S)

    def _has_audio(self):
        return self.frames_np is not None

    def cleanup(self):
        super().cleanup()
        self._flushed.set()


class WindowedSession(_EngineSession):
    """Drives an ``OfflineBackend`` by re-transcribing a window of unprocessed audio."""

    def __init__(self, websocket, backend: OfflineBackend, backend_name, client_uid,
                 language=None, task="transcribe", hotwords=None, initial_prompt=None,
                 max_window_s=15.0, min_chunk_s=1.0, transcribe_opts=None, **kwargs):
        super().__init__(websocket, backend_name, backend.capabilities, client_uid,
                         language=language or backend.DEFAULT_LANGUAGE, task=task, **kwargs)
        self.backend = backend
        self.hotwords = hotwords
        self.initial_prompt = initial_prompt
        self.max_window_s = max_window_s
        self.min_chunk_s = min_chunk_s
        self.transcribe_opts = transcribe_opts or {}
        self._start()

    def add_frames(self, frame_np):
        if self.vad is not None:
            try:
                self.vad.update(frame_np)
            except Exception as e:
                # fail open: unknown audio counts as speech so nothing is silently dropped
                logging.error(f"Session VAD failed, treating frame as speech: {e}")
                self.vad.total_s += len(frame_np) / self.RATE
                self.vad.last_speech_s = self.vad.total_s
        super().add_frames(frame_np)

    # ---- the loop ---------------------------------------------------------------
    def speech_to_text(self):
        while not self.exit:
            if self.frames_np is None:
                self.frames_ready.wait(timeout=self.FIRST_FRAME_WAIT_TIMEOUT_S)
                continue
            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            chunk, duration = self.get_audio_chunk_for_processing()
            if duration > self.max_window_s:
                # a backlog (faster-than-real-time replay, or a stall) is worked through one capped
                # window at a time instead of handing the model a minute of audio at once
                chunk, duration = chunk[: int(self.max_window_s * self.RATE)], float(self.max_window_s)
            flushing = self._flush.is_set()
            if duration < self.min_chunk_s and not (flushing and duration > 0.05):
                if flushing:
                    self._flush.clear()
                    self._flushed.set()
                self.frames_ready.clear()
                self.frames_ready.wait(timeout=self.AUDIO_WAIT_TIMEOUT_S)
                continue

            try:
                if self.vad is not None and not self.vad.speech_since(self.timestamp_offset):
                    # nothing but silence in this window: skip it without calling the model
                    with self.lock:
                        self.timestamp_offset += duration
                    self.frames_ready.clear()
                    self.frames_ready.wait(timeout=self.VOICE_WAIT_TIMEOUT_S)
                    continue

                t0 = time.time()
                segments = self._transcribe(chunk.copy())
                wl_metrics.track_transcription_latency(time.time() - t0)
                wl_metrics.track_audio_processed(duration)
                self._handle(segments, duration, force=flushing)
            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                wl_metrics.track_error("transcription")
                time.sleep(self.ERROR_BACKOFF_S)
            finally:
                if flushing:
                    self._flush.clear()
                    self._flushed.set()

    def _transcribe(self, audio):
        kwargs = dict(self.transcribe_opts)
        if self.hotwords:
            kwargs["hotwords"] = self.hotwords
        if self.initial_prompt:
            kwargs["initial_prompt"] = self.initial_prompt
        lock = getattr(self.backend, "_engine_lock", None)
        if self.backend.NEEDS_LOCK and lock is not None:
            with lock:
                return self.backend.transcribe(audio, language=self.language, task=self.task, **kwargs)
        return self.backend.transcribe(audio, language=self.language, task=self.task, **kwargs)

    def _handle(self, segments, duration, force=False):
        segments = [s for s in (segments or []) if s.text and s.text.strip()]
        self._announce_language(segments)
        with self.lock:
            base = self.timestamp_offset

        # judge the endpoint on the window that was transcribed, not on audio that arrived since:
        # otherwise the last syllables are cut off and come back as a stray one-word segment
        endpoint = bool(self.vad and self.vad.endpoint_at(base, base + duration, self.min_endpoint_silence_s))
        too_long = duration >= self.max_window_s - 1e-6
        if not segments:
            if endpoint or force:
                with self.lock:
                    self.timestamp_offset += duration
            return

        if endpoint or too_long or force:
            for s in segments:
                self._commit(s, base, duration)
            with self.lock:
                self.timestamp_offset += duration
            self.current_out = self.prev_out = ""
            self.same_output_count = 0
            self.end_time_for_same_output = None
            self._trim_transcript()
            out = self.prepare_segments(None)
        else:
            last = self.update_segments(segments, duration)
            out = self.prepare_segments(last)
        if out:
            self.send_transcription_to_client(out)


class StreamingSession(_EngineSession):
    """Drives a ``StreamingBackend``: chunks go straight in, no window is re-transcribed."""

    def __init__(self, websocket, backend: StreamingBackend, backend_name, client_uid,
                 language=None, task="transcribe", max_utterance_s=30.0, **kwargs):
        super().__init__(websocket, backend_name, backend.capabilities, client_uid,
                         language=language or backend.DEFAULT_LANGUAGE, task=task, **kwargs)
        self.backend = backend
        self.max_utterance_s = max_utterance_s
        self._chunks = queue.Queue()
        self.fed_s = 0.0          # audio the backend has consumed
        self.committed_s = 0.0    # session time of the last endpoint
        self._last_sent = None
        self._started_audio = False
        self._start()

    def _has_audio(self):
        return self._started_audio

    def add_frames(self, frame_np):
        self._started_audio = True
        if self.vad is not None:
            try:
                self.vad.update(frame_np)
            except Exception as e:
                logging.error(f"Session VAD failed: {e}")
        self._chunks.put(frame_np)
        self.frames_ready.set()

    def speech_to_text(self):
        while not self.exit:
            try:
                chunk = self._chunks.get(timeout=self.AUDIO_WAIT_TIMEOUT_S)
            except queue.Empty:
                if self._flush.is_set():
                    self._finish_flush()
                continue
            try:
                chunk = self._drain(chunk)
                t0 = time.time()
                self.backend.accept(chunk)
                self.fed_s += len(chunk) / self.RATE
                partial = self.backend.partial()
                wl_metrics.track_transcription_latency(time.time() - t0)
                wl_metrics.track_audio_processed(len(chunk) / self.RATE)

                caught_up = self._chunks.empty()
                vad_end = bool(caught_up and self.vad and
                               self.vad.endpoint(self.committed_s, self.min_endpoint_silence_s))
                too_long = self.fed_s - self.committed_s >= self.max_utterance_s
                if self.backend.is_endpoint() or vad_end or too_long:
                    self._end_utterance()
                elif partial is not None and partial.text.strip():
                    self._announce_language([partial])
                    self._send_partial(partial)
            except Exception as e:
                logging.error(f"[ERROR]: Streaming backend failed: {e}")
                wl_metrics.track_error("transcription")
                time.sleep(self.ERROR_BACKOFF_S)

    def _drain(self, chunk):
        parts = [chunk]
        while True:
            try:
                parts.append(self._chunks.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(parts) if len(parts) > 1 else chunk

    def _send_partial(self, partial):
        text = partial.text.strip()
        if text == self._last_sent:
            return
        self._last_sent = text
        seg = self.format_segment(self.committed_s, max(self.fed_s, self.committed_s), text, completed=False)
        self.send_transcription_to_client(self.prepare_segments(seg))

    def _end_utterance(self):
        final = self.backend.final()
        if final is not None and final.text.strip():
            self._announce_language([final])
            self._commit(final, self.committed_s, self.fed_s - self.committed_s)
            self._trim_transcript()
            self.send_transcription_to_client(self.prepare_segments(None))
        self.backend.reset()
        self.committed_s = self.fed_s
        self._last_sent = None

    def _finish_flush(self):
        if self._chunks.empty():
            self._end_utterance()
        self._flush.clear()
        self._flushed.set()
