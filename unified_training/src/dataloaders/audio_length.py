"""Get audio duration from file path without loading full audio (metadata only)."""
import os
from typing import Optional

_DUMMY_DURATION_SEC = 1.0
_DEFAULT_MISSING_SEC = 1.0


def get_audio_duration_sec(path: str) -> float:
    """
    Get audio duration in seconds from file path.
    Uses metadata only - no full audio read. Instant.
    - Dummy paths (/tmp/dummy...): return _DUMMY_DURATION_SEC
    - File not found: return _DEFAULT_MISSING_SEC
    - Real files: soundfile header (or os.path.getsize fallback)
    """
    if path.startswith("/tmp/dummy"):
        return _DUMMY_DURATION_SEC

    if not os.path.exists(path):
        return _DEFAULT_MISSING_SEC

    try:
        import soundfile as sf
        with sf.SoundFile(path) as f:
            # len(f) = num frames; reads header only
            return len(f) / f.samplerate
    except Exception:
        pass

    # Fallback: file size as rough proxy (no audio read)
    try:
        size_bytes = os.path.getsize(path)
        # Assume ~32kbps mono WAV: ~4000 bytes/sec. Very rough.
        return max(0.1, size_bytes / 4000.0)
    except OSError:
        return _DEFAULT_MISSING_SEC
