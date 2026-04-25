"""
Observation builder for the PCP Arb RL environment.
Converts environment state into natural language for the LLM agent.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def build_text_observation(session_date: str, session_time: str, minutes_to_close: int, daily_pnl: float,
                            positions_count: int, max_positions: int,
                            available_tools: List[str], last_tool_results: Dict,
                            positions_info: List[Dict], violations: List[Dict],
                            risk_utilization: Dict,
                            fundamentals_context: Optional[Dict] = None) -> str:
    """Build natural language observation for the LLM agent.
    
    When fundamentals_context is provided, appends market-wide fundamental
    signals (PE zscore, turbulence, regime, arb propensity) from the NIFTY50
    dataset to give the agent 25 years of context per decision.
    """
    lines = []
    pnl_str = f"+INR {daily_pnl:,.0f}" if daily_pnl >= 0 else f"-INR {abs(daily_pnl):,.0f}"
    lines.append(f"Date: {session_date} | Session: {session_time} | {minutes_to_close} min to close | Daily P&L: {pnl_str} | Positions: {positions_count}/{max_positions}")
    lines.append("")

    if available_tools:
        lines.append(f"Available tools: {', '.join(available_tools)}")
    
    if last_tool_results:
        lines.append("Last tool results:")
        for tool, result in last_tool_results.items():
            res_summary = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            lines.append(f"  {tool}: {res_summary}")
    lines.append("")

    if positions_info:
        lines.append("Current positions:")
        for p in positions_info:
            pnl_val = p.get('pnl', 0)
            pnl_s = f"+{pnl_val:.0f}" if pnl_val >= 0 else f"{pnl_val:.0f}"
            lines.append(f"  {p.get('type', '?')} {p.get('strike', 0):.0f} x {p.get('qty', 0)} | P&L: {pnl_s}")
        lines.append("")

    if risk_utilization:
        lines.append(f"Risk: {risk_utilization.get('margin_used_pct', 0):.1f}% margin used, "
                     f"{risk_utilization.get('concentration_pct', 0):.1f}% max concentration")
    
    lines.append("")
    if violations:
        lines.append("Active violations:")
        for v in violations[:5]:
            lines.append(f"  {v.get('underlying', '?')} {v.get('strike', 0):.0f}: "
                         f"{v.get('deviation_pct', 0):.2f}%, {v.get('trend', 'stable')}, "
                         f"{v.get('active_seconds', 0):.0f}s active")
        lines.append("")

    # --- Fundamentals context (from NIFTY50 25-year dataset) ---
    if fundamentals_context:
        lines.append("Market fundamentals:")
        pe_z = fundamentals_context.get("pe_zscore", 0)
        pe_label = "elevated" if pe_z > 1 else "depressed" if pe_z < -1 else "normal"
        lines.append(f"  NIFTY50 PE zscore {pe_z:+.1f} ({pe_label})")
        em = fundamentals_context.get("earnings_momentum", 0)
        lines.append(f"  Earnings momentum {em:+.1f}%")
        turb = fundamentals_context.get("turbulence", 0)
        turb_label = "high" if turb > 200 else "moderate" if turb > 100 else "low"
        lines.append(f"  Turbulence index {turb:.0f} ({turb_label})")
        regime = fundamentals_context.get("regime", "normal")
        lines.append(f"  Regime: {regime}")
        arb_prop = fundamentals_context.get("arb_propensity", 0.5)
        lines.append(f"  Arb propensity score: {arb_prop:.2f}")
        rec = fundamentals_context.get("ensemble_recommendation", "balanced positioning")
        lines.append(f"  Ensemble recommends: {rec}")
        lines.append("")

    lines.append('Action: output JSON with optional tool_calls and trade action.')
    lines.append('Format: {"tool_calls": [{"server": "...", "tool": "...", "params": {...}}], '
                 '"action_type": "hold|enter_long_call_short_put|enter_short_call_long_put|exit_all|exit_strike", '
                 '"strike": null, "qty": 1}')
    return "\n".join(lines)