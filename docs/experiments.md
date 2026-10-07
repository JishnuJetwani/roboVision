# Curriculum experiments

The controller must learn contact and grip retention before it can learn a full
pickup. Two development comparisons informed the curriculum and update rules.
The [included trial records](../benchmarks/curriculum-trials.json) preserve the
parent checkpoint hashes, interaction budgets, and episode outcomes.

## Getting past 79 mm

Both arm-correction variants started from the same checkpoint and received
204,800 new physical interactions. The difference was whether learned residual
arm corrections were always active or gated by finger opening.

| Measurement | Always active | Gated by finger opening |
| --- | ---: | ---: |
| Initial 79 mm gate: deterministic / stochastic successes | 0/1 · 0/3 | 0/1 · 0/3 |
| First 79 mm gate after training | 0/1 · 0/3 | 1/1 · 3/3 |
| 80 mm gate | Not reached | 1/1 · 3/3 |
| Final frontier | 79 mm | 85 mm |
| Final frontier: deterministic / stochastic successes | 0/1 · 0/10 | 0/1 · 0/10 |
| Final hold-retention check | 1/1 · 10/10 | 1/1 · 10/10 |
| Final pickup-retention check | 1/1 · 10/10 | 1/1 · 10/10 |

Gating made progress while preserving the measured easier skills. It did not
solve the next height jump: the 80-to-85 mm transition still failed. The final
frontier scores refer to different heights and are not a matched success-rate
comparison. Each training gate used one deterministic episode and three stochastic
episodes on repeated development seeds, so these gates are promotion checks,
not independent estimates of generalization.

## Replacing the large height jump

The next curriculum used 1 mm increments. In 102,400 additional interactions it
passed 81, 82, and 83 mm; at the final 84 mm check it achieved 0/1 deterministic
and 1/10 stochastic pickups. A further 102,400 interactions passed 84, 85, and
86 mm, then reached a failing 87 mm check with 0/1 deterministic and 1/10
stochastic pickups. Both runs retained 1/1 deterministic and 10/10 stochastic
successes on each easier retention task.

This supported smaller curriculum steps, while also showing the limits of
extending one pickup controller. The deployed architecture separates approach
and pickup, and trains a manager to choose the handoff. Approach arrival states
provide pickup practice at the states that the preceding policy actually reaches.

## Complete-task evaluation

The selected three-policy controller completed 199/200 pickups across continuous
starting heights from 25 to 140 mm. Among the 78 starts at least 100 mm high,
77 succeeded. The remaining episode ended out of bounds. This is a separate
complete-task measurement; the development gates above are not pooled into it.

[Episode-level evidence](../benchmarks/centered-height-evaluation.json) includes
the exact heights, seeds, handoff timing, and outcomes.
