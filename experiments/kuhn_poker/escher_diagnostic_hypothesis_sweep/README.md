# ESCHER Diagnostic Hypothesis Sweep

This experiment is a quick, single-seed diagnostic sweep for the ESCHER
exploitability-convergence questions. It is intentionally lighter than the
thesis-grade multi-seed experiments and is meant to produce fast directional
evidence.

## Hypotheses Tested

- Removing importance-sampling corrections may better match the ESCHER paper.
- Uniform legal-action fallback at zero positive regret may avoid deterministic
  early-policy bias.
- Combining those two changes may improve exploitability more than either alone.
- Skipping intermediate average-policy training checks whether checkpoint
  evaluation is perturbing final policy extraction.
- Larger author-style Kuhn budgets test whether the current baseline is simply
  too approximation-limited.

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.escher_diagnostic_hypothesis_sweep.run
```

Run a fast subset first:

```bash
python -m experiments.kuhn_poker.escher_diagnostic_hypothesis_sweep.run \
  --variant-ids lightweight_current,no_importance_sampling,uniform_zero_regret_fallback,no_is_uniform_fallback
```

Run only the heavier author-style comparison:

```bash
python -m experiments.kuhn_poker.escher_diagnostic_hypothesis_sweep.run \
  --variant-ids author_public_kuhn_budget,author_budget_no_is_uniform
```

## Outputs

- `variant_summary.csv` -- one row per variant, including recomputed final
  exploitability and last intermediate checkpoint exploitability.
- `checkpoint_curves.csv` -- intermediate checkpoint curves plus one final
  recomputed-policy evaluation row per variant.
- `final_exploitability_by_variant.png`
- `intermediate_exploitability_by_iteration.png`
- `average_policy_value_by_iteration.png`
- `experiment_metadata.json`

