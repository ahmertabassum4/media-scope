# Synthetic Factuality LLM Probe

Samples: `100` synthetic screenshots.
Models: `openai/gpt-5.5, anthropic/claude-sonnet-4.6`.

Prompt: unchanged factuality 4-label prompt from the factuality baseline.
Image input: first 16:9 viewport of each synthetic full-page screenshot.

## Summary

| Scope | Model | N | Shifted to donor | Stayed with base | Shift rate | Base exact 4 | Donor exact 4 | Prediction counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| all | `anthropic/claude-sonnet-4.6` | 100 | 10 | 90 | 0.100 | 0.690 | 0.060 | `{"low": 42, "high": 41, "very low": 15, "very high": 2}` |
| all | `openai/gpt-5.5` | 100 | 14 | 86 | 0.140 | 0.670 | 0.080 | `{"high": 42, "low": 40, "very low": 15, "very high": 3}` |
| high_layout_low_headlines | `anthropic/claude-sonnet-4.6` | 51 | 9 | 42 | 0.176 | 0.745 | 0.098 | `{"high": 40, "low": 8, "very high": 2, "very low": 1}` |
| high_layout_low_headlines | `openai/gpt-5.5` | 51 | 10 | 41 | 0.196 | 0.765 | 0.078 | `{"high": 38, "low": 7, "very high": 3, "very low": 3}` |
| low_layout_high_headlines | `anthropic/claude-sonnet-4.6` | 49 | 1 | 48 | 0.020 | 0.633 | 0.020 | `{"low": 34, "very low": 14, "high": 1}` |
| low_layout_high_headlines | `openai/gpt-5.5` | 49 | 4 | 45 | 0.082 | 0.571 | 0.082 | `{"low": 33, "very low": 12, "high": 4}` |

## Files

- `synthetic-factuality-samples\synthetic_factuality_llm_predictions.csv`
- `synthetic-factuality-samples\synthetic_factuality_llm_summary.csv`
- `synthetic-factuality-samples\synthetic_factuality_llm_raw.jsonl`
- `synthetic-factuality-samples\synthetic_factuality_openrouter_usage.csv`