# logs/

`performance_tracking.csv` is the paper-trading audit trail written by `live/alpaca_rebalance.py`.

It is excluded from git (see `.gitignore`). Keep it locally or in a private store.
To recreate it from scratch, run the rebalance script with `--account PAPER --execute`.
