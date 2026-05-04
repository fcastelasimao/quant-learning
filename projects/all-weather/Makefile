CONDA_RUN = conda run -n allweather python3

.PHONY: test backtest compare-allw rebalance-preview rebalance-execute

test:
	$(CONDA_RUN) -m pytest tests/ -v -m "not integration"

backtest:
	$(CONDA_RUN) main.py

compare-allw:
	$(CONDA_RUN) -m research.compare_allw

rebalance-preview:
	$(CONDA_RUN) -m live.alpaca_rebalance --account $(ACCOUNT) --preview

rebalance-execute:
	$(CONDA_RUN) -m live.alpaca_rebalance --account $(ACCOUNT) --execute
