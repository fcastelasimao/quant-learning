.PHONY: test backtest backtest-sensitivity setup

ENV   ?= base
START ?= 2007-01-01

test:
	conda run -n $(ENV) python -m pytest tests/ -v

backtest:
	conda run -n $(ENV) python main.py --start $(START) --no-plots

backtest-plots:
	conda run -n $(ENV) python main.py --start $(START)

backtest-sensitivity:
	conda run -n $(ENV) python main.py --start $(START) --sensitivity --no-plots

setup:
	conda env create -f environment.yml
