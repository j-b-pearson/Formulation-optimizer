# Pre-registration: formulation optimizer v2

Written and committed **before any v2 benchmark run**: no reporting-seed run, no dev-seed timing run and no v1 reproduction rerun had been started when this file was committed. The code that implements the plan was written and unit-tested (offline) first; the only file under `reports/v2/` at commit time is `run_cost_spread.json`, which evaluates the run-cost formula and runs no campaign.

Any change to a choice below after a reporting-seed result exists is a **deviation**. Deviations are logged in `PROGRESS.md` (section "v2 Decisions", subsection "Deviations from the pre-registration") and stated in the README.

## 0. What stays fixed (v1 settings)

Everything not named below is held at the v1 values in `config.yaml`: the frozen enamel simulator coefficients, prices, specs and oven limit; the concrete emulator and its 50 MPa spec; the BO settings (6 LHS initial points, 4,000 known-feasible candidates per step, Matern 5/2 ARD GP with the v1 kernel bounds, 2 restarts); budgets ($15,000 enamel, $7,500 concrete); common random numbers (noise depends only on seed and step); shared initial designs for all BO variants of a seed; success judged on the noise-free truth. No setting is tuned in v2.

**Success (v1 definition, unchanged):** a campaign succeeds if, before its budget runs out, it runs at least one recipe that is truly in spec (noise-free) and whose material cost is at most 1.05 x the reference optimum (enamel 2.9258 $/kg, so at most 3.0721 $/kg).

## 1. Seeds

| Use | Seeds |
|---|---|
| Reporting (every confirmatory and descriptive v2 run) | 31 to 130 (100 seeds, all new; v1 used 0 to 29 and dev seeds 1000 to 1009) |
| Dev (timing, debugging, STEP 5 only) | 1001 to 1010 |
| v1 reproduction check | 0 to 29 (the v1 seeds, default settings) |

The same seed numbers are used at every spread level and for every method, so all comparisons are paired by seed.

**Runtime rule (from the brief).** Before the full runs, the STEP 1 sweep and the v1-settings run are timed on the 10 dev seeds with `--n-jobs 6` (MLflow logging on, as in the full runs). Estimated total = measured dev wall time x 100 / 10, summed over both runs (`uv run python -m formopt runtime-estimate` writes `reports/v2/runtime_estimate.json`). If the estimate exceeds 6 hours, the reporting seeds are cut to 31 to 90 for every v2 run and this is logged as a deviation. The full runs use the same `--n-jobs 6`.

## 2. The STEP 1 manipulation: run-cost spread s

**Driver.** The v1 enamel run cost is `fixed + driver(x)` with

```
driver(x) = batch_kg * material_cost(x) + energy_rate * (oven_temp - 400) * (30 / line_speed)
```

that is, a trial batch of a pricier resin blend costs more, and a hotter oven and a longer dwell (slower line) cost more energy and line time. v2 keeps this driver and only changes how widely run costs spread.

**Family.**

```
u(x)          = (driver(x) - driver_min) / (driver_max - driver_min)        in [0, 1]
run_cost_s(x) = k_s * (1 + (s - 1) * u(x))
k_s           = mean_v1 / (1 + (s - 1) * mean_u)
```

- **Design space.** The known-feasible box: every input in its v1 range with the oven capped at the 560 C equipment limit. This is the set of runs a lab can actually order; the truly feasible region is unknown to the lab, so it does not define cost.
- **Extremes (exact).** The driver is monotone in each input over this box (material cost rises with PAI fraction, solids and catalyst at the v1 prices; energy rises with temperature and falls with line speed), so its minimum and maximum are at corners. All 32 corners are evaluated: driver_min = 149.0, driver_max = 1747.6. So the cheapest possible run costs k_s and the most expensive k_s x s: **max/min = s exactly**. A unit test also checks 200,000 random points never fall outside these extremes.
- **Budget held fixed.** The budget stays in dollars ($15,000) at every level, as in v1. The mean run cost over the box (uniform distribution) is held at the v1 value, so s changes the spread of run costs, not the overall budget: a uniformly random campaign affords about the same number of runs (about 21) at every level. The mean is a Monte Carlo average over 200,000 uniform known-feasible points (seed 2026): mean_v1 = $711.48, mean_u = 0.25803.
- **v1 effective spread.** The v1 run cost ranges from $299.00 to $1,897.60 over the box (corners), so **v1's effective s = 6.3465**. (v1's calibration report quoted 5.96 because it used the min and max of 200,000 random samples, which never hit the exact corners.) The v1 run cost is exactly the member of this family with s = 6.3465.
- **Constants** (`reports/v2/run_cost_spread.json`): s = 1: every run costs $711.48. s = 3: $469.30 to $1,407.90. s = 10: $214.16 to $2,141.57.
- **Default.** `enamel.run_cost_spread.s: null` keeps the v1 formula unchanged (same operation order, bit for bit), so v1 reproduces with default settings.
- **Expected control at s = 1.** All runs cost the same, so dividing the acquisition by run cost cannot change which candidate wins: bo_cost and bo_cost_cooled are expected to pick exactly the same runs as bo_plain at s = 1 (a unit test checks this on a short campaign). Whatever the data show at s = 1 is reported; the decision rules below are applied unchanged.

## 3. Runs

| Run | What | Seeds | Output |
|---|---|---|---|
| R0a | v1 reproduction: default settings, both problems, all 6 v1 methods, `--n-jobs 6`, compared exactly with `reports/v1` (wall-clock fields excluded) | 0 to 29 | `reports/v2/v1_repro/` |
| R0b | Timing runs of R1 and R2 | 1001 to 1010 | `reports/v2/dev/` |
| R1 | STEP 1 sweep: enamel at s = 1, 3, 10; methods `random`, `lhs`, `bo_plain`, `bo_cost`, `bo_cost_cooled` | 31 to 130 | `reports/v2/step1_sweep/s_<s>/` |
| R2 | v1 settings (default config, v1 run cost): enamel and concrete, all 6 v1 methods | 31 to 130 | `reports/v2/v1_settings/` |
| R5 | STEP 5 exploratory batch comparison | 1001 to 1010 | `reports/v2/step5_batch_dev/` |

If R0a does not reproduce v1 exactly, no other run starts until the cause is found and logged.

Commands: `v2-run --what sweep`, `v2-run --what v1-settings` (add `--dev` for R0b), `benchmark --out-dir reports/v2/v1_repro --no-mlflow --n-jobs 6 --problem enamel concrete` then `repro-check`, `v2-analyze`, `tolerance-curves`, `batch-compare`.

## 4. H1 (STEP 1, main experiment)

**Hypothesis H1.** Cost-aware BO's advantage over plain constrained BO, measured as cumulative spend to first reach spec plus success within budget, grows with the run-cost spread s.

**Headline method:** `bo_cost_cooled` (v1's pre-chosen headline). `bo_cost` is analysed the same way as a secondary method and is not part of the H1 verdict. Baseline: `bo_plain`.

**Co-primary metrics** (per seed, then averaged over the 100 seeds):

1. **Spend saved to first reach spec.** "Reach spec" means reaching the project's target: the first experiment whose recipe is truly in spec and within 5 % of the reference optimum, which is v1's spend-to-target. A campaign that never gets there is **censored at the budget** ($15,000, a lower bound on its true spend, the same rule as v1). Advantage `A_spend = spend(bo_plain) - spend(cost-aware)`, in dollars; positive means the cost-aware method got there for less.
   - Why not "the first in-spec recipe of any cost": on the v1 seeds 0 to 29, `bo_cost_cooled` and `bo_plain` reached their first truly in-spec recipe at the same spend in 28 of 30 seeds (median: the third experiment, inside the 6-point initial design that all BO variants share), so that reading cannot see any difference between acquisition functions. It is reported as secondary metric S2.
2. **Success gain.** `A_success = success(cost-aware) - success(bo_plain)`, each 0 or 1 (v1 success definition).

**Per level.** For each s, the mean advantage over seeds with a 95 % percentile bootstrap CI (resampling seeds; 2,000 resamples, seed 0, the same settings as v1).

**Growth statistic.** `G = mean over seeds of [A(s = 10) - A(s = 1)]`, using per-seed differences (same seed numbers at both levels), with a 95 % percentile bootstrap CI over seeds (2,000 resamples, seed 0).

**Criterion, per co-primary metric:** the CI of G lies entirely above zero **and** the point estimates do not decrease across the levels: A(1) <= A(3) <= A(10).

**Verdict:** H1 is **supported** if both co-primary metrics meet the criterion, **partially supported** if exactly one does, **not supported** if neither does.

**"Where cost-awareness starts to pay" (for the README).** The smallest tested s at which the headline method's per-level CI lies entirely above zero for at least one co-primary metric while the other co-primary point estimate is not negative. If no tested level qualifies, the README says cost-awareness never paid within s <= 10.

**Secondary and descriptive (no verdict):**

- S1: final regret reduction (`final_regret(bo_plain) - final_regret(cost-aware)`), per level, with CI.
- S2: spend saved to the first truly in-spec recipe of any cost, censored at the budget (needs the long CSVs).
- S3: mechanism check per level and method: mean run cost per experiment, number of experiments, success rate, median and mean final regret. Expectation, not a test: cost-aware methods run cheaper experiments as s grows.
- The whole H1 analysis for `bo_cost`.
- The v1-settings point (R2 enamel, s = 6.3465) is drawn on the heterogeneity figure as a hollow reference marker. It is not part of the H1 test.
- Spend vs best-found-cost curves at each s (mean over seeds, 95 % bootstrap band).

## 5. H2 (STEP 2, confirmatory test of fixed-alpha cost-aware BO)

**Hypothesis H2.** `bo_cost` (fixed alpha = 1) beats plain constrained BO (`bo_plain`) on enamel success rate at default v1 settings. `bo_cost` is the headline method for this test only.

**Data.** R2 enamel, seeds 31 to 130, paired by seed.

**Metric.** Per-seed success difference `success(bo_cost) - success(bo_plain)`; its mean with a 95 % percentile bootstrap CI (2,000 resamples, seed 0).

**Decision.** H2 **replicates** if the lower CI bound is above zero. Otherwise the README says plainly that it did not replicate. Also reported: both success rates, the discordant seed counts (only `bo_cost` succeeded / only `bo_plain` succeeded) and an exact two-sided McNemar (binomial) p-value.

**Context.** v1 found a success gain of +0.233 [0.033, 0.433] for `bo_cost` over `bo_plain` on seeds 0 to 29, but `bo_cost` was not the pre-chosen method then, so that was a post-hoc observation. H2 tests it on new seeds. Rough power: with v1's discordance rate (13 of 30 seeds), 100 seeds give a standard error of about 0.063 for the paired difference, so a true gain of about 0.18 or more would be detected with roughly 80 % power; a smaller true gain may well be missed.

## 6. Multiplicity and robustness

- Two confirmatory hypotheses (H1 on `bo_cost_cooled`, H2 on `bo_cost`), each with its own 95 % decision rule. No family-wide correction across them; each is reported on its own terms.
- H1 "supported" requires both co-primary metrics to pass (an intersection rule), which needs no alpha adjustment.
- Per-level CIs (3 levels x 2 metrics), the "starts to pay" level, `bo_cost` under H1 and every secondary metric are unadjusted and descriptive.
- **Robustness re-check:** every verdict and the "starts to pay" level are recomputed with 10,000 resamples (seed 1). If any changes, both are reported, the pre-registered result (2,000 resamples, seed 0) stays the verdict, and it is flagged as fragile.
- No data exclusion: every seed of every run is analysed. A crashed campaign stops the run; nothing is dropped silently.

## 7. STEP 3 (descriptive, no new claims)

Concrete success rate against tolerance from the reference optimum (2, 5, 10 and 15 %) at the full budget, and against experiment budget (number of trial batches) at each tolerance, for every method, with 95 % Wilson CIs. Data: v1 concrete runs (seeds 0 to 29, `reports/v1/benchmark_concrete.csv`) pooled with R2 concrete (seeds 31 to 130), only if both use the same reference optimum; per-source curves are written too. A campaign that ran out of budget keeps its final state at larger experiment budgets (concrete run costs vary by about 2 %, so experiment budget and dollar budget are nearly the same thing). These curves are labelled descriptive.

## 8. STEP 4 (engineering, no hypothesis)

The risk label is computed in code from the probability of passing all specs, with thresholds in `config.yaml` (`lab_note.risk_thresholds`: below 0.5 High, 0.5 to below 0.8 Medium, 0.8 or above Low). The LLM writes sentences around the label and numbers the code supplies. A note whose label differs, that does not quote the pass probability, that uses another label word in the risk sentence, or that contains any number not among the supplied facts is rejected; after one retry a template note is used. Offline tests use a mocked LLM. One live check with `qwen3:14b` is saved to `reports/v2/lab_note_check_enamel.json`.

## 9. STEP 5 (exploratory, dev seeds only)

A q = 4 kriging-believer batch mode (4 experiments per oven cycle; after each pick the GP posterior is updated with its own predicted mean, hyperparameters fixed). Compared with sequential BO for `bo_plain` and `bo_cost_cooled` at v1 settings on dev seeds 1001 to 1010 only: success rate, final regret, experiments and oven cycles. Labelled exploratory; no claim is made from it.

## 10. Order of operations

1. Commit this file.
2. R0a (v1 reproduction). Stop if it fails.
3. R0b (dev timing), runtime estimate, apply the seed rule.
4. R1 and R2 on the reporting seeds.
5. `v2-analyze` (H1, H2, robustness, figures), `tolerance-curves`, STEP 4 live check, R5.
6. README, WALKTHROUGH, V2_REPORT from the files written by the code.

Bug policy: if a bug that affects results is found after reporting-seed results exist, it is fixed, every affected run is repeated on the same seeds, and the fix is logged as a deviation with both results reported.
