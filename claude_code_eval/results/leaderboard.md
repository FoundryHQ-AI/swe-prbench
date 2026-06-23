# Claude Code CLI on SWE-PRBench — Results

Evaluated on the `eval_100` split, judged by Claude Sonnet 4.6 via the local `claude` CLI. 4 agent configurations.

## Headline leaderboard

| Rank | Agent | Overall (mean A/B/C) | DR_A | FPR_A | Halluc_A | F1_A | Coverage_A |
|------|-------|---------------------|------|-------|----------|------|------------|
| 1 | `opus_4_8_low` | 0.437 | 0.633 | 0.080 | 0.080 | 0.500 | 0.860 |
| 2 | `opus_4_8_max` | 0.436 | 0.642 | 0.072 | 0.072 | 0.496 | 0.880 |
| 3 | `opus_4_8_medium` | 0.434 | 0.665 | 0.082 | 0.082 | 0.517 | 0.870 |
| 4 | `opus_4_7_max` | 0.411 | 0.696 | 0.068 | 0.068 | 0.447 | 0.890 |

## Per-context overall score

| Agent | config_A (diff only) | config_B (with file content) | config_C (full context) |
|-------|----------------------|------------------------------|------------------------|
| `opus_4_8_low` | 0.466 | 0.414 | 0.430 |
| `opus_4_8_max` | 0.467 | 0.416 | 0.425 |
| `opus_4_8_medium` | 0.479 | 0.410 | 0.412 |
| `opus_4_7_max` | 0.452 | 0.394 | 0.387 |

## Run-level diagnostics

| Agent | Judge | PRs | Records | Parse failures | Judge fallbacks |
|-------|-------|-----|---------|----------------|-----------------|
| `opus_4_8_low` | `claude_sonnet_46_judge` | 100 | 300 | 0 | 0 |
| `opus_4_8_max` | `claude_sonnet_46_judge` | 100 | 300 | 0 | 1 |
| `opus_4_8_medium` | `claude_sonnet_46_judge` | 100 | 300 | 0 | 1 |
| `opus_4_7_max` | `claude_sonnet_46_judge` | 100 | 300 | 0 | 2 |

