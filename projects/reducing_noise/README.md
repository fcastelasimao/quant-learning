# Reducing Noise

Small SPY research project: test whether denoising a price curve improves next-day return prediction or a simple long/cash trading rule.

This is deliberately a research harness, not a trading strategy. The main danger is look-ahead bias: a full-sample FFT can make a beautiful smooth curve because it has already seen the future. All performance metrics here use rolling causal filters and a one-day execution lag.

## Methods

- Raw return baseline: today direction predicts tomorrow.
- Moving average and exponential moving average baselines.
- FFT denoising: keep the largest frequency components by retained energy or top-k magnitude inside each rolling window.
- SSA/POD-style denoising: build a lagged trajectory matrix, run SVD, keep dominant components, then reconstruct by diagonal averaging.

POD is basically the same family as PCA/SVD denoising. For one time series, Singular Spectrum Analysis is the more natural form because it turns one curve into a lagged matrix before applying SVD.

## Run

```bash
cd projects/reducing_noise
python main.py
```

Outputs are written to `results/`:

- `denoised_series.png`
- `denoised_series.html`
- `parameter_sweep.csv`
- `backtest_summary.csv`
- `equity_curves.png`
- `equity_curves.html`
- `fft_component_split.png`
- `fft_component_split.html`
- `fft_support_resistance.png`
- `fft_support_resistance.html`
- `fft_touch_events.csv`
- `fft_support_resistance_summary.csv`
- `fft_accuracy_over_time.csv`
- `fft_accuracy_over_time.png`
- `fft_accuracy_over_time.html`
- `research_notes.md`

Data is cached under `data/spy_prices.csv`. The default start date is SPY inception, `1993-01-29`.

## Interpretation Rules

- Treat the full-sample denoised line as a visualization only.
- Inspect the FFT residual instead of assuming discarded modes are useless.
- Treat the FFT support/resistance markers as hypotheses until the event summary and over-time accuracy confirm them.
- Prefer methods that improve out-of-sample metrics across several nearby parameters.
- Be suspicious of tiny Sharpe gains with high turnover.
- Remember that high-frequency components are not automatically noise; they can contain the signal you are trying to trade.

## References

- Fourier denoising for index forecasting: https://doaj.org/article/f31e10a6c6494a98a4540c25b8c17bde
- SSA stock denoising and deep learning: https://www.mdpi.com/2813-2203/5/1/9
- Wavelet/SSA denoising with LSTM: https://arxiv.org/abs/2103.03505
- Principal components for stock prediction: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230124
- Mode decomposition caution on high-frequency components: https://peerj.com/articles/cs-1852/
