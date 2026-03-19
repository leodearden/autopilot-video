"""Tests for autopilot.analyze.captions — selective video captioning module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
