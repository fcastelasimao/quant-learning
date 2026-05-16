CONDA_RUN = conda run -n allweather python

.PHONY: test backtest compare-allw compare-fmp-rp leverage-comparison leverage-oos-validation mixed-leverage mixed-leverage-oos-validation mixed-leverage-sweep notebook-data notebook-comparison notebook-leverage rebalance-preview rebalance-execute

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

mixed-leverage-sweep:
	$(CONDA_RUN) -m research.validate_mixed_leverage_oos --sweep-depth disciplined

notebook-data:
	conda run -n allweather marimo edit notebooks/data_explorer.py

notebook-comparison:
	conda run -n allweather marimo edit notebooks/strategy_comparison.py

notebook-leverage:
	conda run -n allweather marimo edit notebooks/leverage_comparison.py

rebalance-preview:
	$(CONDA_RUN) -m live.alpaca_rebalance --paper --account $(ACCOUNT)

rebalance-execute:
	$(CONDA_RUN) -m live.alpaca_rebalance --paper --account $(ACCOUNT) --execute
