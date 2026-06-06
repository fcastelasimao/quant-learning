# Personal Projects

Quantitative finance research and backtesting projects. Each project explores a different strategy or asset class, built from scratch in Python.

> **Disclaimer:** This is educational and research software, not financial advice. Past performance does not guarantee future results.

> **Note:** The **all-weather** risk-parity portfolio graduated out of this repo into the `victecs` product repo (`victecs/all-weather/`). Its full history remains in this repo's git log (`git log -- projects/all-weather`).

---

## Projects

| Project | Status | Description |
|---------|--------|-------------|
| [vol-surface](projects/vol-surface/) | Starting | Volatility surface construction, Heston/SABR calibration, exotic pricing |
| [stat-arb](projects/stat-arb/) | Starting | Cross-sectional equity factor model and long/short backtesting |
| [wave-rider](projects/wave-rider/) | Active | Momentum + regime-based tactical cross-asset strategy |
| [qframe](projects/qframe/) | Active | Factor research harness with sealed hold-out validation |
| [reducing_noise](projects/reducing_noise/) | Active | Signal de-noising / filtering experiments |
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
personal_projects/
├── projects/
│   ├── vol-surface/          Options pricing & vol surface engine
│   ├── stat-arb/             Equity factor model & statistical arbitrage
│   ├── wave-rider/           Cross-asset trend strategy
│   ├── qframe/               Factor research harness
│   ├── reducing_noise/       Signal de-noising experiments
│   └── funding-rate-arb/     Funding rate arb (planned)
├── archive/                  Concluded projects
├── notes/                    Learning notes and snippets
├── resources/                Reference material
└── roadmaps/                 Project planning
```

## Shared data engine

Market data is provided by the workspace-level **`quantcore`** package (`../quantcore`),
which resolves the shared SQLite store (`QuantFinance/data/`) and exposes the FMP
ingestion CLI (`quantcore-ingest`). Projects that need market data install it editable:
`pip install -e ../../quantcore`.

## Quick Start

Each project has its own conda environment (see each project's `environment.yml`):

```bash
# Wave Rider
cd projects/wave-rider && python3 main.py
cd projects/wave-rider && python3 -m pytest tests/ -v
```

## License

This repository is for personal research and education.
