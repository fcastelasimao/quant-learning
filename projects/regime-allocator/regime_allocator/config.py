from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StrategyConfig:
    universe: Dict[str, str] = field(default_factory=lambda: {
        "SPY": "US Large Cap",
        "EFA": "Intl Developed",
        "EEM": "Emerging Markets",
        "TLT": "Long Treasuries",
        "IEF": "Intermediate Treasuries",
        "TIP": "TIPS",
        "GLD": "Gold",
        "DBC": "Broad Commodities",
        "VNQ": "Real Estate",
    })
    start_date: str = "2007-01-01"
    n_regimes: int = 2
    min_train_days: int = 504
    vol_lookback: int = 21
    corr_lookback: int = 63
    max_position: float = 0.40
    min_position: float = 0.0
    risk_aversion: float = 1.0
    cost_bps: float = 10.0
    turnover_buffer: float = 0.03
    drawdown_cutback: float = 0.25

    @property
    def tickers(self):
        return list(self.universe.keys())

    @property
    def asset_names(self):
        return list(self.universe.values())
