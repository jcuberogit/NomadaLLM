"""
NomadaLLM EEG Streamer

Synthetic 8-channel EEG signal generator with neuro-privacy enforcement.
Generates state-modulated epochs (focus/relax) and strips biometric
fingerprints before any data is exposed to the LLM.

Dependency: numpy (required), mne (optional — for real-device pipelines).
Security: Raw EEG never leaves this module. Only anonymised band-power
          features are returned for LLM consumption.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nomadallm.privacy.layer import PrivacyLayer, PrivacyProcessingResult
from nomadallm.privacy.modes import PrivacyMode
from nomadallm.privacy.audit import AuditLogger


# ── Constants ────────────────────────────────────────────────────────────────

CHANNELS: Tuple[str, ...] = ("Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2")
NUM_CHANNELS: int = len(CHANNELS)

SAMPLE_RATE: int = 256          # Hz — standard BCI sampling rate
EPOCH_DURATION: float = 1.0     # seconds per epoch

# Canonical EEG frequency bands (Hz)
BANDS: Dict[str, Tuple[float, float]] = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha":  (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Base amplitude (µV) per band — resting-state baseline
_BASE_AMP: Dict[str, float] = {
    "delta": 20.0,
    "theta": 10.0,
    "alpha": 15.0,
    "beta":   5.0,
    "gamma":  2.0,
}

# State modulation multipliers
_STATE_GAIN: Dict[str, Dict[str, float]] = {
    "focus": {
        "delta": 0.6, "theta": 0.8, "alpha": 0.7,
        "beta":  2.5, "gamma": 1.8,
    },
    "relax": {
        "delta": 1.0, "theta": 1.2, "alpha": 2.8,
        "beta":  0.5, "gamma": 0.4,
    },
    "neutral": {
        "delta": 1.0, "theta": 1.0, "alpha": 1.0,
        "beta":  1.0, "gamma": 1.0,
    },
}

VALID_STATES = tuple(_STATE_GAIN.keys())

# Entropy thresholds for epoch quality gate (spectral entropy, 0-1 normalised)
ENTROPY_FLOOR: float = 0.15   # below → flat-line / disconnection
ENTROPY_CEIL:  float = 0.92   # above → muscle artifact / movement noise


class EpochRejected(Exception):
    """Raised when an epoch fails the entropy sanity check."""

    def __init__(self, entropy: float, reason: str):
        self.entropy = entropy
        self.reason = reason
        super().__init__(f"Epoch rejected ({reason}): entropy={entropy:.4f}")


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class EEGEpoch:
    """A single time-window of multi-channel EEG data.

    Attributes:
        data: numpy array of shape (n_channels, n_samples).
        channels: Channel labels.
        sample_rate: Sampling frequency in Hz.
        state: Cognitive state used for generation.
        biometric_stripped: Whether the subject fingerprint was removed.
    """
    data: np.ndarray
    channels: Tuple[str, ...]
    sample_rate: int
    state: str
    biometric_stripped: bool = False


@dataclass
class EEGFeatures:
    """Band-power feature vector extracted from an epoch.

    This is the *only* data structure safe to pass to the LLM.
    It contains intention (band powers) but NOT identity (raw signal).

    Attributes:
        band_powers: dict mapping band name → array of shape (n_channels,).
        channels: Channel labels.
        state: Cognitive state used for generation.
        privacy_result: Output from PrivacyLayer processing.
    """
    band_powers: Dict[str, np.ndarray]
    channels: Tuple[str, ...]
    state: str
    privacy_result: Optional[PrivacyProcessingResult] = None

    def to_llm_text(self) -> str:
        """Serialise features into a compact text for the LLM prompt.

        Returns only anonymised band powers — no raw signal, no
        biometric signature.
        """
        lines = [f"state={self.state}"]
        for band, powers in self.band_powers.items():
            vals = ",".join(f"{v:.2f}" for v in powers)
            lines.append(f"{band}=[{vals}]")
        return " | ".join(lines)


# ── EEG Streamer ─────────────────────────────────────────────────────────────

class EEGStreamer:
    """Synthetic 8-channel EEG generator with neuro-privacy enforcement.

    Usage::

        streamer = EEGStreamer(subject_id="subj_001")
        features = streamer.generate_epoch(state="focus")
        safe_text = features.to_llm_text()
        # → "state=focus | delta=[...] | theta=[...] | alpha=[...] | beta=[...] | gamma=[...]"

    The raw signal never leaves this class. Only band-power features,
    with the biometric fingerprint stripped, are returned.

    Args:
        subject_id: Opaque identifier for the synthetic subject.
            Determines the unique noise fingerprint that gets stripped.
        sample_rate: Sampling rate in Hz (default 256).
        epoch_duration: Duration of each epoch in seconds (default 1.0).
        privacy_mode: PrivacyMode for the internal PrivacyLayer.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        subject_id: str = "anonymous",
        sample_rate: int = SAMPLE_RATE,
        epoch_duration: float = EPOCH_DURATION,
        privacy_mode: PrivacyMode = PrivacyMode.ZERO_KNOWLEDGE,
        seed: Optional[int] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._subject_id = subject_id
        self._user_hash = hashlib.sha256(subject_id.encode()).hexdigest()[:12]
        self._sr = sample_rate
        self._epoch_len = int(sample_rate * epoch_duration)
        self._privacy = PrivacyLayer(mode=privacy_mode)
        self._audit = audit_logger or AuditLogger()

        # Derive a per-subject RNG seed from the subject_id
        id_hash = int(hashlib.sha256(subject_id.encode()).hexdigest()[:8], 16)
        self._subject_seed = id_hash
        self._rng = np.random.default_rng(seed if seed is not None else id_hash)

        # Pre-generate the subject's biometric fingerprint (static noise
        # signature unique to this "brain"). This is what we strip.
        fp_rng = np.random.default_rng(self._subject_seed)
        self._fingerprint = fp_rng.normal(
            loc=0.0, scale=1.0, size=(NUM_CHANNELS, self._epoch_len)
        ).astype(np.float32)

        # Track last state for audit delta
        self._last_state: Optional[str] = None

        # Optional: mne integration flag
        self._mne_available = self._probe_mne()

    # ── public API ───────────────────────────────────────────────────────

    def generate_epoch(self, state: str = "focus") -> EEGFeatures:
        """Generate one epoch of synthetic EEG and return anonymised features.

        Args:
            state: Cognitive state — ``'focus'``, ``'relax'``, or
                ``'neutral'``.

        Returns:
            EEGFeatures with band powers and PrivacyLayer result.
            Raw signal is never exposed.

        Raises:
            ValueError: Unknown state.
            EpochRejected: Epoch failed the entropy sanity check.
        """
        if state not in VALID_STATES:
            raise ValueError(
                f"Unknown state '{state}'. Valid: {VALID_STATES}"
            )

        # Step 1: Synthesise raw epoch (signal + biometric fingerprint)
        raw = self._synthesise_raw(state)

        # Step 2: Strip biometric fingerprint
        clean = self._strip_fingerprint(raw)

        # Step 3: Entropy sanity check — reject noise / disconnection
        entropy = self._spectral_entropy(clean)
        self._enforce_entropy_gate(entropy, state)

        # Step 4: Extract band-power features (intention, not identity)
        band_powers = self._extract_band_powers(clean)

        # Step 5: Serialise features and run through PrivacyLayer
        features = EEGFeatures(
            band_powers=band_powers,
            channels=CHANNELS,
            state=state,
        )
        privacy_result = self._privacy.process(
            text=features.to_llm_text(),
            user_id=self._user_hash,
            context={"source": "eeg_streamer", "state": state},
        )
        features.privacy_result = privacy_result

        # Step 6: Neuro audit trail (state + hash, never raw signal)
        bp_hash = hashlib.sha256(
            features.to_llm_text().encode()
        ).hexdigest()[:16]
        self._audit.log_neuro_state(
            user_id=self._user_hash,
            state=state,
            band_power_hash=bp_hash,
            entropy=entropy,
        )
        self._last_state = state

        return features

    def get_channel_names(self) -> Tuple[str, ...]:
        """Return the 10-20 channel labels."""
        return CHANNELS

    def get_sample_rate(self) -> int:
        """Return the sampling rate in Hz."""
        return self._sr

    # ── entropy gate (private) ────────────────────────────────────────────

    def _spectral_entropy(self, signal: np.ndarray) -> float:
        """Compute normalised spectral entropy across all channels.

        Returns a value in [0, 1]. Low → narrow-band / flat-line.
        High → broadband noise / muscle artifact.
        """
        n = signal.shape[1]
        psd = np.abs(np.fft.rfft(signal, axis=1)) ** 2
        # Normalise PSD to probability distribution per channel
        psd_sum = psd.sum(axis=1, keepdims=True)
        psd_sum = np.where(psd_sum == 0, 1.0, psd_sum)  # avoid div-by-zero
        p = psd / psd_sum
        # Shannon entropy per channel, normalised by max possible
        log_p = np.where(p > 0, np.log2(p), 0.0)
        h = -np.sum(p * log_p, axis=1) / np.log2(psd.shape[1])
        return float(h.mean())

    def _enforce_entropy_gate(self, entropy: float, state: str) -> None:
        """Reject epoch if entropy falls outside acceptable range."""
        if entropy < ENTROPY_FLOOR:
            reason = "flat_line_or_disconnection"
            self._audit.log_neuro_epoch_rejected(
                user_id=self._user_hash,
                reason=reason,
                entropy=entropy,
            )
            raise EpochRejected(entropy, reason)

        if entropy > ENTROPY_CEIL:
            reason = "muscle_artifact_or_noise"
            self._audit.log_neuro_epoch_rejected(
                user_id=self._user_hash,
                reason=reason,
                entropy=entropy,
            )
            raise EpochRejected(entropy, reason)

    # ── signal synthesis (private) ───────────────────────────────────────

    def _synthesise_raw(self, state: str) -> np.ndarray:
        """Build a (n_channels, n_samples) array of synthetic EEG.

        Each channel = sum of band oscillations + biometric fingerprint.
        """
        n = self._epoch_len
        t = np.linspace(0.0, n / self._sr, n, endpoint=False, dtype=np.float32)
        gains = _STATE_GAIN[state]

        signal = np.zeros((NUM_CHANNELS, n), dtype=np.float32)

        for band_name, (flo, fhi) in BANDS.items():
            amp = _BASE_AMP[band_name] * gains[band_name]
            # Centre frequency + small per-channel jitter
            fc = (flo + fhi) / 2.0
            for ch in range(NUM_CHANNELS):
                freq = fc + self._rng.uniform(-0.5, 0.5)
                phase = self._rng.uniform(0.0, 2.0 * np.pi)
                signal[ch] += amp * np.sin(2.0 * np.pi * freq * t + phase)

        # Add biometric fingerprint (scaled to ~3 µV — subtle but unique)
        signal += self._fingerprint * 3.0

        # Add measurement noise (~1 µV Gaussian)
        signal += self._rng.normal(0.0, 1.0, signal.shape).astype(np.float32)

        return signal

    def _strip_fingerprint(self, raw: np.ndarray) -> np.ndarray:
        """Remove the subject's biometric noise signature.

        Subtracts the pre-generated fingerprint so the residual carries
        cognitive-state information but NOT individual identity.
        """
        return raw - (self._fingerprint * 3.0)

    def _extract_band_powers(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute mean band power per channel via FFT.

        Returns dict mapping band name → array of shape (n_channels,).
        Units: µV² (power spectral density estimate).
        """
        n = signal.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / self._sr)
        fft_mag = np.abs(np.fft.rfft(signal, axis=1)) ** 2 / n

        powers: Dict[str, np.ndarray] = {}
        for band_name, (flo, fhi) in BANDS.items():
            mask = (freqs >= flo) & (freqs < fhi)
            if mask.any():
                powers[band_name] = fft_mag[:, mask].mean(axis=1)
            else:
                powers[band_name] = np.zeros(NUM_CHANNELS, dtype=np.float32)

        return powers

    # ── mne integration (optional) ───────────────────────────────────────

    @staticmethod
    def _probe_mne() -> bool:
        """Check if mne-python is available."""
        try:
            import mne  # noqa: F401
            return True
        except ImportError:
            return False

    def to_mne_raw(self, epoch_data: np.ndarray):
        """Convert a raw epoch array to an mne.io.RawArray.

        Only available if ``mne`` is installed.

        Args:
            epoch_data: Array of shape (n_channels, n_samples).

        Returns:
            mne.io.RawArray with standard 10-20 montage.

        Raises:
            ImportError: mne not installed.
        """
        if not self._mne_available:
            raise ImportError(
                "mne-python not installed. Run: pip install mne"
            )
        import mne

        info = mne.create_info(
            ch_names=list(CHANNELS),
            sfreq=self._sr,
            ch_types="eeg",
        )
        # mne expects data in Volts; our data is in µV
        raw = mne.io.RawArray(epoch_data * 1e-6, info)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
        return raw
