# Quant Learning

Quantitative finance research and backtesting projects. Each project explores a different strategy or asset class, built from scratch in Python.

> **Disclaimer:** This is educational and research software, not financial advice. Past performance does not guarantee future results.

---

## Projects

| Project | Status | Description |
|---------|--------|-------------|
| [all-weather](projects/all-weather/) | Active | Risk-parity multi-asset portfolio with Alpaca paper trading |
| [vol-surface](projects/vol-surface/) | Starting | Volatility surface construction, Heston/SABR calibration, exotic pricing |
| [stat-arb](projects/stat-arb/) | Starting | Cross-sectional equity factor model and long/short backtesting |
| [wave-rider](projects/wave-rider/) | Active | Momentum + regime-based tactical cross-asset strategy |
| [funding-rate-arb](projects/funding-rate-arb/) | Planned | Delta-neutral crypto funding rate arbitrage |

### Archived

| Project | Reason |
|---------|--------|
| [pairs-trading](archive/pairs-trading/) | Survivorship bias invalidated backtest results |
| [crypto-cex-arb](archive/crypto-cex-arb/) | Edge below commission drag across tested exchanges |
| [HMM](archive/HMM/) | Regime detection absorbed into wave-rider |

---

## Repository Structure

```
quant-learning/
├── projects/
│   ├── all-weather/          Risk-parity portfolio engine
│   ├── vol-surface/          Options pricing & vol surface engine
│   ├── stat-arb/             Equity factor model & statistical arbitrage
│   ├── wave-rider/           Cross-asset trend strategy
│   └── funding-rate-arb/     Funding rate arb (planned)
├── shared/                   Shared utilities (future)
├── archive/                  Concluded projects
├── notes/                    Learning notes and snippets
├── resources/                Reference material
└── roadmaps/                 Project planning
```

## Quick Start

Both active projects share the `allweather` conda environment:

```bash
# Setup
conda create -n allweather python=3.11
conda activate allweather
pip install -r projects/all-weather/requirements.txt

# All Weather — run backtest
cd projects/all-weather && python3 main.py

# All Weather — compare against Bridgewater ALLW ETF
cd projects/all-weather && python3 compare_allw.py

# Wave Rider — run backtest
cd projects/wave-rider && python3 main.py

# Run tests
cd projects/all-weather && python3 -m pytest tests/ -v
cd projects/wave-rider && python3 -m pytest tests/ -v
```

## License

This repository is for personal research and education.
