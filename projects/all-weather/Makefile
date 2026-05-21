CONDA_RUN = conda run -n allweather python
ACCOUNT ?= PAPER

.PHONY: test backtest compare-allw compare-fmp-rp leverage-comparison leverage-oos-validation mixed-leverage mixed-leverage-oos-validation mixed-leverage-oos-validation-ibkr-safe mixed-leverage-oos-validation-strict-pilot mixed-leverage-oos-validation-ibkr-safe-funding-plus-2 mixed-leverage-oos-validation-ibkr-safe-funding-plus-4 mixed-leverage-oos-validation-ibkr-safe-fmp mixed-leverage-sweep mixed-leverage-sweep-ibkr-safe mixed-leverage-sweep-strict-pilot mixed-leverage-sweep-ibkr-safe-funding-plus-2 mixed-leverage-sweep-ibkr-safe-funding-plus-4 mixed-leverage-full-grid-oos-ibkr-safe notebook-data notebook-comparison notebook-leverage rebalance-preview rebalance-live-preview rebalance-execute

test:
	$(CONDA_RUN) -m pytest tests/ -v

backtest:
	$(CONDA_RUN) main.py

compare-allw:
	$(CONDA_RUN) -m research.compare_allw

compare-fmp-rp:
	$(CONDA_RUN) -m research.compare_fmp_rp

leverage-comparison:
	$(CONDA_RUN) -m research.build_leverage_comparison_report

leverage-oos-validation:
	$(CONDA_RUN) -m research.validate_leverage_oos

mixed-leverage:
	$(CONDA_RUN) -m research.build_mixed_leverage_report

mixed-leverage-oos-validation:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos

mixed-leverage-oos-validation-ibkr-safe:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --max-global-cap 0.30 --max-sleeve-weight 0.30

mixed-leverage-oos-validation-strict-pilot:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --max-global-cap 0.20 --max-sleeve-weight 0.20

mixed-leverage-oos-validation-ibkr-safe-funding-plus-2:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --max-global-cap 0.30 --max-sleeve-weight 0.30 --funding-spread 0.02

mixed-leverage-oos-validation-ibkr-safe-funding-plus-4:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --max-global-cap 0.30 --max-sleeve-weight 0.30 --funding-spread 0.04

mixed-leverage-oos-validation-ibkr-safe-fmp:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --max-global-cap 0.30 --max-sleeve-weight 0.30 --data-source fmp --fmp-price-column adj_close

mixed-leverage-sweep:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --sweep-depth disciplined

mixed-leverage-sweep-ibkr-safe:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --sweep-depth disciplined --max-global-cap 0.30 --max-sleeve-weight 0.30

mixed-leverage-sweep-strict-pilot:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --sweep-depth disciplined --max-global-cap 0.20 --max-sleeve-weight 0.20

mixed-leverage-sweep-ibkr-safe-funding-plus-2:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --sweep-depth disciplined --max-global-cap 0.30 --max-sleeve-weight 0.30 --funding-spread 0.02

mixed-leverage-sweep-ibkr-safe-funding-plus-4:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --sweep-depth disciplined --max-global-cap 0.30 --max-sleeve-weight 0.30 --funding-spread 0.04

mixed-leverage-full-grid-oos-ibkr-safe:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --full-grid-oos --max-global-cap 0.30 --max-sleeve-weight 0.30

notebook-data:
	conda run -n allweather marimo edit notebooks/data_explorer.py

notebook-comparison:
	conda run -n allweather marimo edit notebooks/strategy_comparison.py

notebook-leverage:
	conda run -n allweather marimo edit notebooks/leverage_comparison.py

rebalance-preview:
	$(CONDA_RUN) -m live.alpaca_rebalance --paper --account $(ACCOUNT) --use-live-tickers

rebalance-live-preview:
	$(CONDA_RUN) -m live.alpaca_rebalance --live --account $(ACCOUNT) --use-live-tickers

rebalance-execute:
	$(CONDA_RUN) -m live.alpaca_rebalance --paper --account $(ACCOUNT) --use-live-tickers --execute
