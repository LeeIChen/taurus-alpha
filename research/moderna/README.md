# Moderna (MRNA) valuation model

Program-level risk-adjusted NPV (sum of the parts) and a P/E cross-check behind the
Oct 2, 2026 research note (Sell, $27 price target at a $188.94 share price).

## Files

| File | What it does |
|---|---|
| `mrna_model.py` | The model: every program's stage, probability of success (with its basis), launch year, peak sales, margin, Moderna share and remaining development cost; balance sheet inputs; DCF and P/E functions. Run it directly for the base-case breakdown. |
| `mrna_analysis.py` | Runs base, bear and bull scenarios, sensitivities (discount rate, melanoma peak sales, overhead), the implied-expectations solve, and writes `mrna_results.json`. |
| `build_note.py` + `note_template.html` | Builds the HTML research note (`mrna_note.html`) from `mrna_results.json`, so no number in the note is typed by hand. |
| `mrna_results.json`, `mrna_note.html` | Outputs from the published run. |

## Re-run

Standard library only; any Python 3.9+ works.

```bash
python research/moderna/mrna_model.py      # base-case program breakdown
python research/moderna/mrna_analysis.py   # scenarios, sensitivities -> mrna_results.json
python research/moderna/build_note.py      # -> mrna_note.html
```

## Changing assumptions

- **Programs:** edit the `Program(...)` entries in `build_programs()` in `mrna_model.py`.
  Intismeran peak sales are global; `share=0.5` applies the Merck 50/50 split.
- **Market and balance sheet:** `PRICE`, `DILUTED_SHARES`, `CASH_PRO_FORMA`, the convertible
  terms, the litigation contingency and `P_LITIGATION_LOSS` are at the top of `mrna_model.py`.
- **Overhead and discount rate:** `corporate_flows()` and the `r` argument of `valuation()`.
- **Scenarios:** the `bear` and `bull` overrides in `mrna_analysis.py`.

Probabilities start from BIO / Informa / QLS, *Clinical Development Success Rates
2011–2020*, adjusted per program; each program's `pos_basis` string records the reasoning.
All inputs are as of Oct 1, 2026.
