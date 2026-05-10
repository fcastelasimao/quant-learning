CONDA_RUN = conda run -n allweather python

.PHONY: test backtest compare-allw compare-fmp-rp notebook-data notebook-comparison rebalance-preview rebalance-execute

test:
	$(CONDA_RUN) -m pytest tests/ -v

backtest:
	$(CONDA_RUN) main.py

compare-allw:
	$(CONDA_RUN) -m research.compare_allw

compare-fmp-rp:
	$(CONDA_RUN) -m research.compare_fmp_rp

notebook-data:
	conda run -n allweather marimo edit notebooks/data_explorer.py

notebook-comparison:
	conda run -n allweather marimo edit notebooks/strategy_comparison.py

rebalance-preview:
	$(CONDA_RUN) -m live.alpaca_rebalance --paper --account $(ACCOUNT)

rebalance-execute:
	$(CONDA_RUN) -m live.alpaca_rebalance --paper --account $(ACCOUNT) --execute
