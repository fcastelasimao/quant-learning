# Visualisation Strategy (post-market-validation)

## Principle

Every chart must answer one question a human actually asks.
Don't visualise data — visualise decisions.

**New principle post-ALLW:** Every chart is both an analytical tool AND
a marketing asset. Design every chart to be shareable from day one.
If it doesn't make someone stop scrolling, it's not ready.

---

## Three chart categories

### Category 1: Marketing charts (highest priority)
These exist to attract users and validate demand. They are designed for
Reddit, Twitter/X, blog posts, and the landing page. They must work as
standalone images without surrounding text.

### Category 2: Boss / investor presentation charts
These exist to convince your boss, co-founders, or potential investors
that the strategy is sound. They work in a 10-minute presentation.

### Category 3: Client dashboard charts
These exist to serve paying users. They are interactive, personalised,
and focused on their specific portfolio. Built last.

---

## Category 1: Marketing charts

These are the charts that test demand. They must be:
- Shareable as standalone images (no context needed)
- Branded with your product URL/name in a corner
- Designed to provoke reaction: surprise, curiosity, or "I need this"
- High resolution, dark theme, professional quality

### M1: ALLW vs DIY — Drawdown comparison ← #1 PRIORITY
**Question it answers:** "Can I beat ALLW without paying 0.85%?"

- Overlaid drawdown curves: your 6asset_tip_gsg vs ALLW vs 60/40 vs SPY
- Time period: March 2025 (ALLW inception) to present
- Annotate the Iran war start (Feb 28) and key escalations
- Show the key numbers directly on chart: MaxDD for each strategy
- If your strategy has a shallower drawdown than ALLW during the Iran
  war, this single image sells the product.

**Design notes:**
- Dark background (#0d1117), consistent with existing theme
- Your strategy in blue (#58a6ff), ALLW in red (#f85149), 60/40 dotted
- Large, readable annotations — this will be viewed at phone-screen size
- Brand URL in bottom-right corner, subtle but visible
- Title: "Drawdown during the Iran war: DIY vs ALLW (no leverage)"

**Format:** PNG at 1200×675 (Twitter/X optimal), also at 1080×1080 (Instagram/Reddit).

### M2: Fee drag over time — the compound cost of ALLW
**Question it answers:** "How much do ALLW's fees cost me over 20 years?"

- Two lines diverging over time: $100k invested in DIY (0.12% cost) vs
  ALLW (0.85% cost), assuming identical gross returns.
- Annotate the dollar difference at 10, 20, and 30 years.
- At 7% gross return, the difference on $100k over 20 years is roughly
  $15,000-$20,000 in fees alone. This number shocks people.
- This is the simplest chart to produce and one of the most effective.

**Design notes:**
- Two colors only. Clean, minimal. Let the diverging lines speak.
- Show dollar amounts, not percentages — "$18,400 lost to fees" hits harder
  than "0.73% annual drag."

### M3: Four-quadrant regime heatmap
**Question it answers:** "Does this actually work in all environments?"

- 2×2 grid: Growth ↑/↓ × Inflation ↑/↓
- Each cell: strategy name + average monthly return + hit rate (% positive months)
- Color: green for positive, red for negative, intensity = magnitude
- Include ALLW if sufficient data exists (note small sample caveat)
- This is the Dalio thesis in one image. If every cell is green, it works.

**Design notes:**
- Large, bold numbers in each cell. The chart is the data.
- Subtitle: "Average monthly return in each economic environment, 2006-2026"
- This is the most intellectually interesting chart — it will get engagement
  from the quant/finance community on Twitter/X and Hacker News.
- Cannot be produced until regime analysis (experiment R1) is complete.

### M4: Iran war crisis performance — bar chart
**Question it answers:** "How did each strategy handle the current crisis?"

- Horizontal bars showing Feb 28 to present return for each strategy
- Order from best to worst
- Include: your four candidates, ALLW, 60/40, SPY
- Simple, dramatic, timely.

**Design notes:**
- Green bars for positive returns, red for negative
- Publish as quickly as possible after the data is available — timeliness
  is the main value of this chart. It becomes stale in weeks.

---

## Category 2: Boss / investor presentation charts

These are for internal use and pitch decks. They don't need to be
shareable standalone but must be clear in a presentation context.

### P1: Drawdown comparison timeline (full history)
**Question:** "Will this protect me in a crash?"

- X-axis: 2006-2026. Y-axis: % below all-time high.
- One line per strategy + 60/40 + SPY.
- Annotate: 2008 GFC, 2020 COVID, 2022 rate shock, 2026 Iran war.
- SPY drops 50%, 60/40 drops 26%, your strategy drops 20%.
- Visually unmistakable.

### P2: Recovery time comparison bars
**Question:** "How long until I recover if I invest at the worst time?"

- Horizontal bars: max recovery months per strategy.
- SPY took 4+ years to recover from 2008. Your strategy: ~18 months.
- This is the behavioural argument in one chart.

### P3: Monte Carlo fan chart
**Question:** "What could realistically happen to my money?"

- Median outcome line with 10th-90th percentile bands.
- Overlay: actual historical path.
- Requires bootstrap analysis (experiment R2) as input.

### P4: Risk-adjusted return scatter
**Question:** "Where does this sit on the risk-return spectrum?"

- X: Max Drawdown. Y: CAGR. Size: Calmar ratio.
- SPY upper-right (high return, high risk). Your strategies center.
- Include ALLW as a point for direct comparison.
- Existing scatter_calmar_mdd.png is a starting point — refine it.

### P5: Walk-forward validation chart (already exists)
**Question:** "How do I know this isn't overfit?"

- Already produced by validation.py. The existing chart is good.
- For presentations, add a one-line caption explaining what it means.
- This is your key differentiator — ALLW doesn't publish anything like this.

---

## Category 3: Client dashboard

Built after demand is validated (after Gate 2 in experiment plan).
Interactive, personalised, mobile-first.

### D1: Portfolio value over time
Line chart: selected strategy vs 60/40 vs SPY vs ALLW.
Toggle strategies on/off. Zoom into date ranges.
Mobile: swipeable, not zoomable.

### D2: Current allocation vs target
Donut chart: current weights vs target weights.
Green = within threshold, amber = drifting, red = rebalance needed.
Mobile: simple vertical bars may work better than donuts.

### D3: Monthly rebalancing instructions
Table: Ticker, Action, Dollar Amount, Shares.
This is the core product deliverable. Must be perfectly clear on mobile.
"You need to buy 3 shares of PDBC and sell 2 shares of TLT."

### D4: Performance summary card
CAGR, MaxDD, Calmar, Ulcer in large readable numbers.
Comparison to ALLW and 60/40 underneath each metric.
Mobile: card layout, one metric per row.

### D5: Regime indicator
Traffic light: current economic regime.
"Current: Rising Inflation + Falling Growth (Stagflation).
Your commodity and inflation bond allocations are working."
Updates monthly based on SPY momentum + TIP-TLT spread.

---

## Design system

### Theme: dark, professional, differentiated
Keep the existing dark theme. It looks professional and immediately
differentiates from every white-background competitor.

**Palette:**
- Background: #0d1117
- Surface: #161b22
- Text primary: #c9d1d9
- Text secondary: #8b949e
- Blue (your strategies): #58a6ff
- Green (positive/60-40): #3fb950
- Coral (SPY): #f78166
- Amber (buy & hold): #f0b429
- **Red (ALLW): #f85149** ← new, distinct from SPY coral
- Purple (alternative strategies): #d2a8ff

### Colour mapping for ALLW comparisons
In any chart that includes ALLW:
- Your strategy: blue (#58a6ff) — the hero
- ALLW: red (#f85149) — the incumbent challenger
- 60/40: green (#3fb950) — the baseline
- SPY: coral (#f78166) — the raw-return reference

This creates a consistent visual language: blue = us, red = them.

### Branding on shareable charts
Every marketing chart (Category 1) includes:
- Product URL in bottom-right corner (small, 8pt, #8b949e)
- Chart title as part of the image (not just matplotlib title)
- No matplotlib toolbar or figure chrome — clean export only

### Mobile-first for all dashboards
- Min touch target: 44×44px
- Font size: never below 14px on mobile
- Charts: use line charts over bar charts (better at small sizes)
- Tables: max 4 columns on mobile. Scroll horizontally for more.
- Donut charts: replace with horizontal stacked bars on small screens

### Two output formats for every marketing chart
1. **1200×675px** — Twitter/X and blog embeds (16:9)
2. **1080×1080px** — Reddit and Instagram (square)

Produce both automatically. Use matplotlib `savefig` with explicit
figsize and dpi. Add padding for the branding watermark.

---

## Implementation priority (updated)

| Priority | Chart | Category | Effort | Depends on |
|----------|-------|----------|--------|-----------|
| **1** | **ALLW vs DIY drawdown** | Marketing | 2 hours | B0 experiment |
| **2** | **Fee drag over time** | Marketing | 1 hour | Simple math |
| **3** | **Iran war crisis bars** | Marketing | 1 hour | B1 experiment |
| 4 | Full-history drawdown timeline | Presentation | 2 hours | None |
| 5 | Recovery time comparison | Presentation | 1 hour | None |
| 6 | Regime heatmap | Marketing + Presentation | 3 hours | R1 experiment |
| 7 | Risk-adjusted scatter (with ALLW) | Presentation | 1 hour | B0 data |
| 8 | Monte Carlo fan chart | Presentation | 3 hours | R2 bootstrap |
| 9 | Interactive client dashboard | Client | 8-12 hours | After Gate 2 |

The top 3 charts can all be produced within days of completing the ALLW
comparison experiment (B0) and the Iran stress test (B1). They form the
content package for the blog post, landing page, and Reddit launch.

---

## Implementation notes

### Where the code lives
- Marketing charts: new module `marketing_charts.py` (separate from
  `plotting.py` which serves the backtest output). Marketing charts
  have different requirements: branding, dual-format export, no
  matplotlib chrome.
- Presentation charts: extend `plotting.py` with new functions.
- Client dashboard: new module or extend `results_dashboard.py`.

### Automation
After the first manual production, automate chart generation so that
every new backtest run produces updated marketing + presentation charts.
Add to the experiment pipeline: after full_backtest completes, generate
all charts that have the data they need.

### Quality gate
Before publishing any chart:
1. Can I understand it in 5 seconds? (If no → simplify)
2. Is the key number visible on the chart? (If no → annotate)
3. Does it work at phone screen size? (If no → enlarge text/simplify)
4. Is the brand URL present? (If no → add watermark)
5. Would I share this if I saw it? (If no → make it more surprising)