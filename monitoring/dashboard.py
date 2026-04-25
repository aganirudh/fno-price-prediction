"""
Rich-based live terminal dashboard for monitoring the PCP Arb system.
Displays market state, agent actions, positions, PnL, and training metrics.
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live

from config.settings import get_settings, LOGS_DIR

class Dashboard:
    """Rich-based live terminal dashboard with 8 panels."""

    def __init__(self):
        self.console = Console()
        self.settings = get_settings()
        self._market_state: Dict = {}
        self._agent_state: Dict = {}
        self._positions: List[Dict] = []
        self._session_pnl: float = 0.0
        self._equity_points: List[float] = []
        self._training_metrics: Dict = {}
        self._feed_health: Dict = {}
        self._fundamentals_state: Dict = {}
        self._ensemble_state: Dict = {}
        self._step_count = 0
        self._log_file: Optional[Path] = None
        self._log_entries: List[Dict] = []

    def start_logging(self, session_name: str = None):
        """Enable logging of dashboard updates to a JSONL file."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = session_name or f"session_{ts}"
        self._log_file = LOGS_DIR / f"{name}.jsonl"

    def update(self, market: Dict = None, agent: Dict = None, positions: List[Dict] = None,
               pnl: float = None, training: Dict = None, feed_health: Dict = None,
               step: int = None, fundamentals: Dict = None, ensemble: Dict = None):
        """Update dashboard state."""
        if market: self._market_state = market
        if agent: self._agent_state = agent
        if positions is not None: self._positions = positions
        if pnl is not None:
            self._session_pnl = pnl
            self._equity_points.append(pnl)
        if training: self._training_metrics = training
        if feed_health: self._feed_health = feed_health
        if step is not None: self._step_count = step
        if fundamentals: self._fundamentals_state = fundamentals
        if ensemble: self._ensemble_state = ensemble
        entry = {
            "timestamp": datetime.now().isoformat(), "step": self._step_count,
            "pnl": self._session_pnl, "positions": len(self._positions),
            "market": self._market_state, "agent": self._agent_state,
            "training": self._training_metrics
        }
        if self._log_file:
            import json
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _build_market_panel(self) -> Panel:
        """Panel 1 - Market State: Active violations and underlyings."""
        table = Table(box=None, expand=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Strike", justify="right")
        table.add_column("Dev %", justify="right", style="bold yellow")
        table.add_column("Trend", justify="center")

        violations = self._market_state.get("violations", [])
        for v in violations[:6]:
            trend_map = {"up": "[green]+[/green]", "down": "[red]-[/red]", "stable": "[dim]=[/dim]"}
            table.add_row(
                v.get("underlying", "?"),
                f"{v.get('strike', 0):.0f}",
                f"{v.get('deviation_pct', 0):.2f}%",
                trend_map.get(v.get("trend", "stable"), "=")
            )
        
        if not violations:
            return Panel(Text("No violations detected", style="dim center"), title="Market State")
        return Panel(table, title="Market State")

    def _build_agent_panel(self) -> Panel:
        """Panel 2 - Agent Activity: Last action, reasoning, and tool calls."""
        lines = []
        action = self._agent_state.get("action", {})
        action_type = action.get("action_type", "hold")
        color = "green" if "enter" in action_type else "red" if "exit" in action_type else "dim"
        lines.append(f"Last Action: [{color}]{action_type}[/{color}]")
        
        if "strike" in action and action["strike"]:
            lines.append(f"Strike: {action['strike']}")
        
        tool_calls = self._agent_state.get("tool_calls", [])
        if tool_calls:
            lines.append(f"Tools: [cyan]{', '.join([tc.get('tool','') for tc in tool_calls])}[/cyan]")
        
        reward = self._agent_state.get("reward_breakdown", {})
        if reward:
            lines.append(f"Total Reward: [bold]{reward.get('total', 0):.4f}[/bold]")

        return Panel("\n".join(lines), title="Agent Activity")

    def _build_positions_panel(self) -> Panel:
        """Panel 3 - Current Positions."""
        table = Table(box=None, expand=True)
        table.add_column("ID", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Strike", justify="right")
        table.add_column("Qty", justify="right")
        table.add_column("PnL", justify="right")

        for i, p in enumerate(self._positions[:5]):
            pnl = p.get("pnl", 0)
            pnl_style = "green" if pnl >= 0 else "red"
            table.add_row(
                str(i+1), p.get("type", "?"), str(p.get("strike", 0)),
                str(p.get("qty", 0)), f"[{pnl_style}]{pnl:+.0f}[/{pnl_style}]"
            )
        
        if not self._positions:
            return Panel(Text("No open positions", style="dim center"), title="Portfolio")
        return Panel(table, title=f"Portfolio ({len(self._positions)} pos)")

    def _build_pnl_panel(self) -> Panel:
        """Panel 4 - PnL & Equity Curve (Sparkline)."""
        pnl = self._session_pnl
        color = "green" if pnl >= 0 else "red"
        pnl_text = Text(f"INR {pnl:,.0f}", style=f"bold {color} font_size=20")
        
        # Simple sparkline logic
        if len(self._equity_points) > 2:
            points = self._equity_points[-20:]
            min_p, max_p = min(points), max(points)
            range_p = max_p - min_p if max_p > min_p else 1
            spark = ""
            chars = " _.-'^"
            for p in points:
                idx = int((p - min_p) / range_p * (len(chars)-1))
                spark += chars[idx]
            spark_text = Text(f"\n\nEquity: {spark}", style="cyan")
            return Panel(pnl_text + spark_text, title="Session PnL")
        
        return Panel(pnl_text, title="Session PnL")

    def _build_training_panel(self) -> Panel:
        """Panel 5 - Training Metrics (GRPO)."""
        lines = []
        m = self._training_metrics
        if m:
            lines.append(f"Steps: {m.get('step', 0)}")
            lines.append(f"Avg Reward: [bold green]{m.get('avg_reward', 0):.4f}[/bold green]")
            lines.append(f"Policy Loss: {m.get('loss', 0):.4f}")
            lines.append(f"KL Div: {m.get('kl', 0):.4f}")
        else:
            lines.append("[dim]No training metrics[/dim]")
        return Panel("\n".join(lines), title="Training")

    def _build_health_panel(self) -> Panel:
        """Panel 6 - System Health: MCP servers and latency."""
        lines = []
        h = self._feed_health
        if h:
            for server, status in h.items():
                icon = "[green]OK[/green]" if status.get("alive") else "[red]ERR[/red]"
                latency = status.get("latency_ms", 0)
                lines.append(f"{server:12}: {icon} ({latency}ms)")
        else:
            lines.append("[dim]No health data[/dim]")
        return Panel("\n".join(lines), title="System Health")

    def _build_fundamentals_panel(self) -> Panel:
        """Panel 7 - Fundamentals: PE zscore, turbulence, regime, arb propensity."""
        lines = []
        f = self._fundamentals_state
        if f:
            pe_z = f.get("pe_zscore", 0)
            pe_color = "red" if pe_z > 1.5 else "yellow" if pe_z > 0.5 else "green"
            lines.append(f"PE Zscore: [{pe_color}]{pe_z:+.2f}[/{pe_color}]")
            turb = f.get("turbulence", 0)
            turb_color = "red" if turb > 200 else "yellow" if turb > 100 else "green"
            lines.append(f"Turbulence: [{turb_color}]{turb:.0f}[/{turb_color}]")
            regime = f.get("regime", "normal")
            lines.append(f"Regime: [bold cyan]{regime}[/bold cyan]")
            arb = f.get("arb_propensity", 0)
            arb_color = "green" if arb > 0.7 else "yellow" if arb > 0.4 else "dim"
            lines.append(f"Arb Propensity: [{arb_color}]{arb:.2f}[/{arb_color}]")
            active = f.get("active_agent", "ensemble")
            lines.append(f"Active Agent: [bold]{active}[/bold]")
        else:
            lines.append("[dim]No fundamentals loaded[/dim]")
        return Panel("\n".join(lines), title="Fundamentals", border_style="bright_blue")

    def _build_ensemble_panel(self) -> Panel:
        """Panel 8 - Ensemble performance: rolling Sharpe, selection, returns."""
        lines = []
        e = self._ensemble_state
        if e:
            for agent_name in ["PPO", "A2C", "DDPG"]:
                sharpe = e.get(f"sharpe_{agent_name}", 0)
                color = "green" if sharpe > 0.5 else "yellow" if sharpe > 0 else "red"
                lines.append(f"{agent_name} Sharpe: [{color}]{sharpe:.3f}[/{color}]")
            selected = e.get("selected_agent", "?")
            lines.append(f"\nSelected: [bold yellow]{selected}[/bold yellow]")
            next_reb = e.get("next_rebalance", "N/A")
            lines.append(f"Next Rebalance: {next_reb}")
            port_ret = e.get("portfolio_return", 0)
            arb_ret = e.get("arb_return", 0)
            lines.append(f"Portfolio: {port_ret:+.2f}%  Arb: {arb_ret:+.2f}%")
        else:
            lines.append("[dim]Ensemble not active[/dim]")
        return Panel("\n".join(lines), title="Ensemble", border_style="bright_magenta")

    def render(self) -> Layout:
        """Build the full dashboard layout with 8 panels."""
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=12),
            Layout(name="middle", size=10),
            Layout(name="bottom", size=8),
            Layout(name="extra", size=8))
        layout["top"].split_row(
            Layout(self._build_market_panel(), name="market"),
            Layout(self._build_agent_panel(), name="agent"))
        layout["middle"].split_row(
            Layout(self._build_positions_panel(), name="positions"),
            Layout(self._build_pnl_panel(), name="pnl"))
        layout["bottom"].split_row(
            Layout(self._build_training_panel(), name="training"),
            Layout(self._build_health_panel(), name="health"))
        layout["extra"].split_row(
            Layout(self._build_fundamentals_panel(), name="fundamentals"),
            Layout(self._build_ensemble_panel(), name="ensemble"))
        return layout

    def run_live(self, env=None, refresh_rate: float = 1.0):
        """Run the dashboard in a live update loop."""
        with Live(self.render(), console=self.console, refresh_per_second=refresh_rate) as live:
            try:
                import time
                while True:
                    live.update(self.render())
                    time.sleep(1.0/refresh_rate)
            except KeyboardInterrupt:
                pass