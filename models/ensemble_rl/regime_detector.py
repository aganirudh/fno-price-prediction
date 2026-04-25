"""
RegimeDetector - detects market regime from fundamentals + turbulence.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional
import numpy as np
import pandas as pd

from data_pipeline.kaggle.fundamentals_processor import FundamentalsSnapshot

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    label: str
    preferred_agent: str
    arb_propensity: float
    recommended_position_size_multiplier: float
    reasoning: str

class RegimeDetector:
    def __init__(self, pe_overvalued_threshold: float = 1.5, pe_undervalued_threshold: float = -1.0, turbulence_high: float = 150, turbulence_crisis: float = 250):
        self.pe_overvalued = pe_overvalued_threshold
        self.pe_undervalued = pe_undervalued_threshold
        self.turb_high = turbulence_high
        self.turb_crisis = turbulence_crisis

    def detect_regime(self, fundamentals: FundamentalsSnapshot, turbulence: float) -> MarketRegime:
        pe_z = fundamentals.nifty50_pe_zscore
        if turbulence > self.turb_crisis:
            return MarketRegime("crisis", "a2c", 0.9, 0.5, "Crisis conditions")
        if pe_z > self.pe_overvalued and turbulence > self.turb_high:
            return MarketRegime("overvalued_volatile", "a2c", 0.8, 0.7, "Overvalued volatile")
        if pe_z < self.pe_undervalued and turbulence < 100:
            return MarketRegime("undervalued_stable", "ppo", 0.4, 1.2, "Undervalued stable")
        return MarketRegime("normal", "ensemble", 0.6, 1.0, "Normal market")
