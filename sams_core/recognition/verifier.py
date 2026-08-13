"""Collects a student's captured signatures over time and reports any
that don't match the rest, per the coursework's "Recognition" bonus
requirement.

Owned by: Recognition role.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from sams_core.recognition.feature_matcher import OrbSignatureMatcher
