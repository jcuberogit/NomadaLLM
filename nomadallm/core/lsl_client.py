"""
NomadaLLM LSL Client — Universal BCI Ingestion Layer

Connects NomadaLLM to any Lab Streaming Layer (LSL) source:
OpenBCI, Muse, g.tec, ANTneuro, MNE simulators, or custom streams.

Architecture:
    ┌────────────┐     pylsl      ┌──────────────────┐    privacy    ┌─────────┐
    │  EEG Device │──── LSL ─────→│ NomadaLSLReceiver │────gate─────→│ EEGFeat │
    │  / Simulator│   (LAN only)  │  circular buffer  │  + entropy   │ → LLM   │
    └────────────┘                └──────────────────┘              └─────────┘

Dependencies: pylsl (pip install pylsl), numpy.
Security:
    - Streams resolved ONLY on localhost / LAN (RFC 1918).
    - SovereigntyError on external IP detection.
    - Raw samples never leave this module — only EEGFeatures cross the boundary.

Latency target: < 5 ms from LSL pull to feature extraction.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from nomadallm.exceptions import SovereigntyError
from nomadallm.privacy.audit import AuditLogger
from nomadallm.privacy.layer import PrivacyLayer
from nomadallm.privacy.modes import PrivacyMode
from nomadallm.providers.bci.streamer import (
    BANDS,
    CHANNELS as DEFAULT_CHANNELS,
    ENTROPY_CEIL,
    ENTROPY_FLOOR,
    NUM_CHANNELS as DEFAULT_NUM_CHANNELS,
    EEGFeatures,
    EpochRejected,
)

logger = logging.getLogger("nomadallm.lsl")

# ── Constants ────────────────────────────────────────────────────────────────

# RFC 1918 private ranges + loopback — the ONLY acceptable source IPs.
_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),      # link-local IPv6
]

DEFAULT_BUFFER_SECONDS: float = 5.0
DEFAULT_EPOCH_SECONDS: float = 1.0
LSL_RESOLVE_TIMEOUT: float = 3.0          # seconds to wait for stream discovery


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class LSLStreamInfo:
    """Metadata about a discovered LSL stream."""
    name: str
    stream_type: str
    channel_count: int
    sample_rate: float
    source_id: str
    hostname: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.stream_type,
            "channels": self.channel_count,
            "sample_rate": self.sample_rate,
            "source_id": self.source_id,
            "hostname": self.hostname,
        }


# ── NomadaLSLReceiver ───────────────────────────────────────────────────────

class NomadaLSLReceiver:
    """Non-blocking LSL ingestion client with circular buffer and
    sovereignty enforcement.

    Usage::

        receiver = NomadaLSLReceiver()
        receiver.connect("OpenBCI_EEG")   # finds stream on LAN
        features = receiver.pull_epoch()  # returns EEGFeatures
        receiver.stop()

    The receiver runs a background thread that continuously pulls
    samples from the LSL inlet into a fixed-size circular buffer.
    ``pull_epoch()`` extracts the latest epoch-window, runs the
    entropy gate and privacy pipeline, and returns anonymous features.

    Args:
        buffer_seconds: How many seconds of data to keep in RAM.
        epoch_seconds: Duration of each epoch window for feature extraction.
        privacy_mode: PrivacyMode for the internal PrivacyLayer.
        audit_logger: Optional shared AuditLogger instance.
        allowed_hosts: Additional hostnames/IPs to whitelist beyond RFC 1918.
    """

    def __init__(
        self,
        buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
        epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
        privacy_mode: PrivacyMode = PrivacyMode.ZERO_KNOWLEDGE,
        audit_logger: Optional[AuditLogger] = None,
        allowed_hosts: Optional[List[str]] = None,
    ) -> None:
        self._buffer_seconds = buffer_seconds
        self._epoch_seconds = epoch_seconds
        self._privacy = PrivacyLayer(mode=privacy_mode)
        self._audit = audit_logger or AuditLogger()
        self._allowed_hosts: List[str] = list(allowed_hosts or [])

        # Runtime state (set on connect)
        self._inlet = None
        self._stream_info: Optional[LSLStreamInfo] = None
        self._n_channels: int = 0
        self._sr: float = 0.0
        self._channel_names: Tuple[str, ...] = ()

        # Circular buffer: deque of (timestamp, sample_array) tuples
        self._buffer: Deque[Tuple[float, np.ndarray]] = deque()
        self._buffer_lock = threading.Lock()
        self._max_samples: int = 0

        # Background ingestion thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Discovery ────────────────────────────────────────────────────────

    @staticmethod
    def discover(timeout: float = LSL_RESOLVE_TIMEOUT) -> List[LSLStreamInfo]:
        """Scan the local network for EEG-type LSL streams.

        Returns:
            List of discovered stream metadata objects.

        Raises:
            ImportError: pylsl not installed.
        """
        pylsl = _import_pylsl()

        results = pylsl.resolve_byprop("type", "EEG", timeout=timeout)
        streams: List[LSLStreamInfo] = []
        for info in results:
            streams.append(LSLStreamInfo(
                name=info.name(),
                stream_type=info.type(),
                channel_count=info.channel_count(),
                sample_rate=info.nominal_srate(),
                source_id=info.source_id(),
                hostname=info.hostname(),
            ))
        return streams

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self, stream_name: str) -> 'NomadaLSLReceiver':
        """Resolve an LSL stream by name and start ingestion.

        Args:
            stream_name: The ``name`` field of the target LSL stream
                (e.g. ``"OpenBCI_EEG"``, ``"Muse"``, ``"MNE_sim"``).

        Returns:
            Self for method chaining.

        Raises:
            SovereigntyError: Stream hostname resolves to a non-local IP.
            RuntimeError: Stream not found within timeout.
            ImportError: pylsl not installed.
        """
        if self._running:
            self.stop()

        pylsl = _import_pylsl()

        # Resolve stream on local network
        logger.info("Resolving LSL stream '%s' …", stream_name)
        results = pylsl.resolve_byprop(
            "name", stream_name, timeout=LSL_RESOLVE_TIMEOUT
        )
        if not results:
            raise RuntimeError(
                f"LSL stream '{stream_name}' not found on local network "
                f"(timeout={LSL_RESOLVE_TIMEOUT}s). "
                f"Ensure the stream is running and discoverable."
            )

        info = results[0]
        hostname = info.hostname()

        # ── Sovereignty check ────────────────────────────────────────
        self._enforce_sovereignty(hostname)

        self._stream_info = LSLStreamInfo(
            name=info.name(),
            stream_type=info.type(),
            channel_count=info.channel_count(),
            sample_rate=info.nominal_srate(),
            source_id=info.source_id(),
            hostname=hostname,
        )
        self._n_channels = info.channel_count()
        self._sr = info.nominal_srate()
        self._channel_names = self._resolve_channel_names(info)

        # Size the circular buffer
        self._max_samples = int(self._buffer_seconds * self._sr)
        self._buffer.clear()

        # Open LSL inlet with minimal buffer on the network side
        self._inlet = pylsl.StreamInlet(
            info,
            max_buflen=int(self._buffer_seconds),
            max_chunklen=0,                  # get samples ASAP
            recover=True,
        )

        # Start background ingestion
        self._running = True
        self._thread = threading.Thread(
            target=self._ingest_loop,
            name="nomada-lsl-ingest",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "Connected to LSL stream '%s' (%d ch @ %.0f Hz) from %s",
            stream_name, self._n_channels, self._sr, hostname,
        )
        return self

    def stop(self) -> None:
        """Stop ingestion and release the LSL inlet."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
            self._inlet = None

        with self._buffer_lock:
            self._buffer.clear()

        logger.info("LSL stream disconnected.")

    @property
    def connected(self) -> bool:
        return self._running and self._inlet is not None

    @property
    def stream_info(self) -> Optional[LSLStreamInfo]:
        return self._stream_info

    # ── Epoch extraction ─────────────────────────────────────────────────

    def pull_epoch(self, state: str = "neutral") -> EEGFeatures:
        """Extract the latest epoch from the buffer, process, and return
        anonymised features.

        This is the single call that bridges LSL → LLM.

        Args:
            state: Cognitive state label to tag the epoch. Can be
                overridden by an external classifier.

        Returns:
            EEGFeatures — safe for LLM consumption.

        Raises:
            RuntimeError: Not connected or buffer underrun.
            EpochRejected: Epoch failed entropy sanity check.
        """
        if not self.connected:
            raise RuntimeError("Not connected to any LSL stream. Call connect() first.")

        epoch_samples = int(self._epoch_seconds * self._sr)

        with self._buffer_lock:
            if len(self._buffer) < epoch_samples:
                raise RuntimeError(
                    f"Buffer underrun: need {epoch_samples} samples "
                    f"but only {len(self._buffer)} available. "
                    f"Wait for the buffer to fill (~{self._epoch_seconds}s)."
                )
            # Extract the latest epoch_samples from the right (most recent)
            recent = list(self._buffer)[-epoch_samples:]

        # Assemble (n_channels, n_samples) array
        signal = np.array(
            [s for _, s in recent], dtype=np.float32
        ).T  # shape: (n_channels, epoch_samples)

        # ── Entropy gate ─────────────────────────────────────────────
        entropy = self._spectral_entropy(signal)
        user_hash = self._user_hash()

        if entropy < ENTROPY_FLOOR:
            reason = "flat_line_or_disconnection"
            self._audit.log_neuro_epoch_rejected(
                user_id=user_hash, reason=reason, entropy=entropy,
            )
            raise EpochRejected(entropy, reason)

        if entropy > ENTROPY_CEIL:
            reason = "muscle_artifact_or_noise"
            self._audit.log_neuro_epoch_rejected(
                user_id=user_hash, reason=reason, entropy=entropy,
            )
            raise EpochRejected(entropy, reason)

        # ── Band-power extraction ────────────────────────────────────
        band_powers = self._extract_band_powers(signal)

        features = EEGFeatures(
            band_powers=band_powers,
            channels=self._channel_names,
            state=state,
        )

        # ── PrivacyLayer pass ────────────────────────────────────────
        privacy_result = self._privacy.process(
            text=features.to_llm_text(),
            user_id=user_hash,
            context={
                "source": "lsl_receiver",
                "stream": self._stream_info.name if self._stream_info else "unknown",
                "state": state,
            },
        )
        features.privacy_result = privacy_result

        # ── Neuro audit (state + hash, never signal) ─────────────────
        bp_hash = hashlib.sha256(
            features.to_llm_text().encode()
        ).hexdigest()[:16]
        self._audit.log_neuro_state(
            user_id=user_hash,
            state=state,
            band_power_hash=bp_hash,
            entropy=entropy,
        )

        return features

    def buffer_fill_ratio(self) -> float:
        """Return how full the buffer is (0.0–1.0)."""
        if self._max_samples == 0:
            return 0.0
        with self._buffer_lock:
            return len(self._buffer) / self._max_samples

    # ── Background ingestion (private) ───────────────────────────────────

    def _ingest_loop(self) -> None:
        """Continuously pull samples from the LSL inlet into the
        circular buffer.  Runs in a daemon thread."""
        while self._running and self._inlet is not None:
            try:
                # Pull a chunk of available samples (non-blocking, 0 timeout)
                samples, timestamps = self._inlet.pull_chunk(
                    timeout=0.0,
                    max_samples=64,
                )
                if timestamps:
                    with self._buffer_lock:
                        for ts, samp in zip(timestamps, samples):
                            self._buffer.append(
                                (ts, np.array(samp, dtype=np.float32))
                            )
                        # Trim to max size
                        while len(self._buffer) > self._max_samples:
                            self._buffer.popleft()
                else:
                    # No data available — yield CPU
                    time.sleep(0.001)
            except Exception as exc:
                logger.warning("LSL ingest error: %s", exc)
                time.sleep(0.01)

    # ── Sovereignty enforcement (private) ────────────────────────────────

    def _enforce_sovereignty(self, hostname: str) -> None:
        """Verify that the stream source is on the local network.

        Raises SovereigntyError if the hostname resolves to a
        non-RFC-1918 IP address.
        """
        # Allow explicit whitelist
        if hostname in self._allowed_hosts:
            return

        # Resolve hostname to IP
        import socket
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            ips = {ai[4][0] for ai in addr_info}
        except socket.gaierror:
            # If we can't resolve, assume it's a local hostname
            logger.warning(
                "Cannot resolve hostname '%s'; assuming local.", hostname,
            )
            return

        for ip_str in ips:
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            is_local = ip.is_loopback or ip.is_link_local or any(
                ip in net for net in _PRIVATE_NETS
            )
            if not is_local:
                raise SovereigntyError(
                    operation=f"lsl_connect({hostname})",
                    backends_checked=[
                        f"ip={ip_str}",
                        "allowed_nets=RFC1918+loopback",
                    ],
                )

    # ── Signal processing (private) ──────────────────────────────────────

    @staticmethod
    def _spectral_entropy(signal: np.ndarray) -> float:
        """Normalised spectral entropy across all channels [0, 1]."""
        psd = np.abs(np.fft.rfft(signal, axis=1)) ** 2
        psd_sum = psd.sum(axis=1, keepdims=True)
        psd_sum = np.where(psd_sum == 0, 1.0, psd_sum)
        p = psd / psd_sum
        log_p = np.where(p > 0, np.log2(p), 0.0)
        h = -np.sum(p * log_p, axis=1) / np.log2(psd.shape[1])
        return float(h.mean())

    def _extract_band_powers(
        self, signal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """FFT-based band-power extraction."""
        n = signal.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / self._sr)
        fft_mag = np.abs(np.fft.rfft(signal, axis=1)) ** 2 / n

        powers: Dict[str, np.ndarray] = {}
        for band_name, (flo, fhi) in BANDS.items():
            mask = (freqs >= flo) & (freqs < fhi)
            if mask.any():
                powers[band_name] = fft_mag[:, mask].mean(axis=1)
            else:
                powers[band_name] = np.zeros(
                    signal.shape[0], dtype=np.float32
                )
        return powers

    # ── Helpers (private) ────────────────────────────────────────────────

    def _user_hash(self) -> str:
        """Deterministic anonymised user ID derived from stream source."""
        source = (
            self._stream_info.source_id
            if self._stream_info
            else "unknown"
        )
        return hashlib.sha256(source.encode()).hexdigest()[:12]

    @staticmethod
    def _resolve_channel_names(info) -> Tuple[str, ...]:
        """Extract channel labels from LSL stream metadata.

        Falls back to generic labels if the stream doesn't provide them.
        """
        try:
            desc = info.desc()
            channels_node = desc.child("channels")
            names: List[str] = []
            ch = channels_node.child("channel")
            while ch.name() == "channel":
                label = ch.child_value("label")
                if label:
                    names.append(label)
                ch = ch.next_sibling("channel")
            if names:
                return tuple(names)
        except Exception:
            pass

        # Fallback: use default 10-20 labels if channel count matches,
        # otherwise generate Chnn labels.
        n = info.channel_count()
        if n == len(DEFAULT_CHANNELS):
            return DEFAULT_CHANNELS
        return tuple(f"Ch{i+1:02d}" for i in range(n))

    def __del__(self) -> None:
        self.stop()

    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        name = self._stream_info.name if self._stream_info else "none"
        return f"NomadaLSLReceiver(stream={name!r}, status={status})"


# ── Module-level helpers ─────────────────────────────────────────────────────

def _import_pylsl():
    """Lazy import pylsl with a clear error message."""
    try:
        import pylsl
        return pylsl
    except ImportError:
        raise ImportError(
            "pylsl is required for LSL connectivity. "
            "Install it with:  pip install pylsl\n"
            "See: https://labstreaminglayer.readthedocs.io/"
        )
