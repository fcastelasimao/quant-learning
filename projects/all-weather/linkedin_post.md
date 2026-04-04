# LinkedIn Post — Final Version

---

I just finished my Maths PhD and did what any reasonable person would do: tried to replicate a $1B hedge fund strategy from scratch.

Bridgewater launched their All Weather ETF (ALLW) in March 2025 — their risk parity strategy in a wrapper: ~2x leveraged bonds, 0.85% annual fee, powered by decades of institutional research.

I spent 3 months building my own version in Python: risk parity optimisation, 100-ETF universe scan, covariance-based weights validated across 3 independent out-of-sample windows, and monthly rebalancing, now paper-trading live via Alpaca.

The results over the same window (since ALLW launch, fee-adjusted):

DIY Risk Parity vs ALLW:
- CAGR: 17.4% vs 19.1% (ALLW ahead by ~1.7%)
- Max Drawdown: -5.7% vs -8.8% (mine is 35% shallower)
- Calmar Ratio: 3.03 vs 2.18 (mine is 39% better)
- Annual Cost: ~$120 vs ~$850 on $100k

ALLW earns slightly more because it levers up bonds ~2x. But it pays for that leverage with deeper drawdowns. Different product, different investor.

I tested adding leverage myself. Every 0.25x of bond leverage:
- Adds ~0.5% CAGR
- Adds ~3% to max drawdown
- Worsens Calmar by ~20%

In a 2022-style rate shock, 2x leverage turns a -16% drawdown into -25%.

The takeaway: Bridgewater's edge isn't the formula — risk parity is well-understood mathematics. Their edge is execution at scale and the institutional conviction to hold through a -25% drawdown without blinking.

For someone managing their own savings, the unleveraged version captures most of the return with much less tail risk.

If you're interested in the methodology or want to discuss, feel free to reach out.

#QuantitativeFinance #RiskParity #AllWeather #Python #PhD

---

# Notes

- **Tone on Bridgewater**: Respectful — "different product, different investor" + "their edge is execution at scale". Not claiming to be better, just showing a specific trade-off.
- **Code sharing**: Gated — "feel free to reach out" instead of public GitHub link. Can share with serious people who DM.
- **Plot**: Attach the two-panel figure (equity curves + April 2025 drawdown zoom).
