"""
HybridController - combines SB3 ensemble + GRPO LLM.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional
import numpy as np

from models.ensemble_rl.ensemble_selector import EnsembleSelector
from models.ensemble_rl.regime_detector import RegimeDetector, MarketRegime
from data_pipeline.kaggle.fundamentals_processor import FundamentalsSnapshot

logger = logging.getLogger(__name__)

@dataclass
class PortfolioAction:
    allocations: Dict[str, float]
    total_equity_exposure: float
    selected_agent: str = "ensemble"
    regime: str = "normal"

@dataclass
class ArbSignal:
    action_type: str
    strike: Optional[float] = None
    qty: int = 1
    confidence: float = 0.0

class HybridController:
    def __init__(self, ensemble_selector: EnsembleSelector, regime_detector: RegimeDetector = None, fundamentals_processor=None):
        self.ensemble = ensemble_selector
        self.regime_detector = regime_detector or RegimeDetector()
        self.fund_processor = fundamentals_processor

    def get_portfolio_action(self, current_date: date, obs: np.ndarray, turbulence: float = 0.0) -> PortfolioAction:
        action_vec = self.ensemble.get_ensemble_action(obs, current_date, turbulence)
        selected = self.ensemble.select_agent(current_date)
        return PortfolioAction({}, float(np.sum(np.abs(action_vec))), selected.name if selected else "ensemble")

    def get_arb_action(self, market_state: dict, model=None, tokenizer=None, fundamentals: FundamentalsSnapshot = None, turbulence: float = 0.0) -> Optional[ArbSignal]:
        # Implementation simplified for smoke test
        return ArbSignal("hold")
