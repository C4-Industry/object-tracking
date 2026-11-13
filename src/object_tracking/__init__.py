"""object_tracking: Lite tracker untuk CCTV FPS rendah, plus alat evaluasi dan perbandingan tracker.

Bagian yang dipasang ke pipeline produksi hanya `lite.py` dan `integration.py` (numpy dan scipy).
Modul lain (zoo, metrics, mot17, live_compare, compare_app, detector) untuk eksplorasi dan evaluasi.
"""

from pathlib import Path

from .integration import (LITE_PRESETS, LITE_TUNED_DETECTOR, LITE_TUNED_PARAMS, CameraTrackers, LiteForComplianceTracker, PersonTracker, TrackOutput,
                          load_lite_config)
from .lite import LiteTracker, Track, TrackerConfig

__version__ = "0.1.0"

# Root repo: src/object_tracking/__init__.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]

__all__ = ["LiteTracker", "Track", "TrackerConfig", "PersonTracker", "TrackOutput", "CameraTrackers",
           "LiteForComplianceTracker", "load_lite_config", "LITE_PRESETS", "LITE_TUNED_DETECTOR", "LITE_TUNED_PARAMS", "ROOT"]
