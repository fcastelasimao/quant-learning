# logs/

`performance_tracking_<mode>_<account>.csv` is the private paper/live audit
trail written by `live/alpaca_rebalance.py`.

It is excluded from git (see `.gitignore`). Keep it locally or in a private store.
To recreate it from scratch, run the rebalance script with `--paper --account PAPER --execute`.
