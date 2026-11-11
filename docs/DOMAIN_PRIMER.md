# Domain Primer

This primer gives an engineer with no chemistry or R&D background the context behind the simulator, the cost model and the benchmark. When a number here is marked **illustrative**, it is a plausible example value, not a fact about any real product. When it is marked with a source, the source is listed at the bottom.

## 1. What "formulation development" looks like

A formulation is a recipe: a set of ingredient amounts (resins, solvents, catalysts, fillers, cement, water) plus process settings (cure temperature, line speed, mixing time). Development is a loop:

1. Pick a recipe and process settings.
2. Make a batch and run it (coat wire on a line, cast concrete cylinders, spray panels).
3. Test it against a specification ("spec"): a list of pass/fail thresholds on measured properties.
4. Decide what to try next.

Each loop costs money and calendar time. The goal is rarely "the best possible property". It is usually **"the cheapest recipe that passes every spec with some margin, found quickly."** That is the problem this project solves.

## 2. Glossary

| Term | Meaning | Why it matters here |
|---|---|---|
| Formulation / recipe | The ingredient amounts, usually as mass fractions or kg per batch | The input vector `x` (with process settings) |
| Process variable | A setting of the manufacturing step, not an ingredient (oven temperature, line speed, cure age) | Also part of `x`; often drives run cost |
| Spec | Pass/fail limits on properties (e.g. breakdown at least X, viscosity inside a window) | Black-box constraints in the optimizer |
| In spec / feasible | Passes every spec | Only feasible formulations count as results |
| Raw material cost | Cost of the ingredients per kg of product (or per m3 for concrete) | The objective to minimize. It is a **known** formula, not a measurement |
| Run cost / cost per experiment | What one development experiment costs: batch materials, line or oven time, energy, technician time, testing | Drives cost-aware acquisition and the budget |
| Response surface | How a property changes as you move the inputs | What the surrogate model learns |
| DOE | Design of experiments: planned sets of runs (factorial, fractional factorial, Latin hypercube, central composite) | Baselines to beat |
| OFAT | One factor at a time | The naive approach. Misses interactions |
| Interaction | The effect of one variable depends on another (catalyst helps more at low oven temperature) | Why OFAT fails and why a flexible model helps |
| Mixture constraint | Ingredient fractions must sum to 100 percent | Changes how you sample and design. Matters for the concrete problem |
| Known constraint | A limit you can check before running (max oven temperature, fractions sum to 1) | Filter candidates directly. Never "learn" these |
| Unknown constraint | A limit you only learn by running (does it pass flexibility?) | Modeled with probability of feasibility |
| Noise / repeatability | The spread you get when you repeat the same recipe | Sets the GP noise level and the minimum meaningful difference |
| Emulator | A model trained on real data, used as a stand-in lab in benchmarks | Lets us run thousands of virtual experiments. Has validity limits |
| Surrogate | The optimizer's own model of the response, refit after each run | Not the same thing as the emulator |

### Magnet wire enamel terms

| Term | Meaning |
|---|---|
| Magnet wire | Copper or aluminum wire coated with a thin polymer insulation ("enamel") used in motor, transformer, and generator windings |
| Enamel | The liquid coating: resin dissolved in solvent at a given solids content, applied in several thin passes and cured in an oven |
| Polyester-imide (PEI) | A common magnet wire resin family, often modified with THEIC (tris(2-hydroxyethyl) isocyanurate) for higher thermal performance |
| Polyamide-imide (PAI) | A tougher, higher temperature resin. Often used as an overcoat on a polyester or polyester-imide basecoat (dual coat) |
| Solids content | Mass percent of non-volatile resin in the enamel. Higher solids means fewer passes or more build per pass, less solvent to burn off, but higher viscosity |
| Viscosity | Resistance to flow (centipoise, cP). Must sit inside a window for the applicator dies or felts to lay down a uniform film |
| Build | Thickness of the cured insulation layer |
| Cure | Solvent removal and crosslinking/imidization in the oven. Undercure gives poor properties; overcure makes the film brittle |
| Cure oven | Multi-zone oven the wire passes through. Zone air temperatures are set; the film's actual temperature history depends on line speed |
| Line speed | How fast the wire moves through the applicator and oven. Faster means more output but less time at temperature (less cure) |
| Dielectric breakdown | Voltage at which the insulation fails, measured with standard test methods (for example twisted pair) |
| Flexibility / elongation and mandrel wrap | Mechanical tests: stretch the wire and wrap it around a mandrel, then look for cracks |
| Adhesion | How well the film stays bonded to the conductor after stretching or snapping |
| Heat shock | Wrap and expose to high temperature, check for cracking |
| Thermal class | A temperature rating (for example class 155, 180, 200) indicating long-term thermal endurance |

## 3. The enamel simulator (synthetic, illustrative)

The domain demo uses a **synthetic** simulator. It is chemically plausible in direction and order of magnitude, not calibrated to any real enamel.

### Variables and ranges

| Variable | Name in code | Range | Unit | Basis |
|---|---|---|---|---|
| PAI fraction of resin blend | `pai_fraction` | 0.0 to 1.0 | fraction | Illustrative. Real dual-coat systems usually apply PAI as a separate overcoat; the simulator collapses that into one blend variable for teaching |
| Solids content | `solids_pct` | 25 to 45 | mass % | Plausible: a high-solids PAI enamel patent claims about 28 to 45 percent solids [S1] |
| Catalyst level | `catalyst_phr` | 0.0 to 1.0 | parts per hundred resin | Illustrative |
| Cure oven temperature | `oven_temp_c` | 420 to 580 | deg C (zone air) | Plausible: the same patent describes oven outlet zones of about 427 to 593 deg C [S1] |
| Line speed | `line_speed_m_min` | 10 to 40 | m/min | Illustrative. The patent's examples ran 56 ft/min (about 17 m/min) [S1] |

**Known (input) constraint:** `oven_temp_c <= max_cure_temp_c` (default 560 deg C, configurable). This is a safety and equipment limit, so candidates above it are never proposed.

### Outputs

| Output | Name | Spec (default) | Behavior the simulator encodes |
|---|---|---|---|
| Breakdown strength proxy | `breakdown_kv` | at least 30 | Rises with degree of cure and build (solids), rises somewhat with PAI fraction, drops with overcure. Gaussian noise, sd about 1.0 |
| Flexibility pass rate | `flex_pass` | at least 0.8 | Fraction of 5 mandrel-wrap specimens that pass. Drops with overcure and very high solids, rises with PAI fraction. Binomial noise (5 trials) |
| Viscosity at 30 deg C | `viscosity_cp` | 2,000 to 15,000 | Log-viscosity rises with solids and PAI fraction. Lognormal noise, about 3 percent. The window comes from the range the patent claims for its enamel [S1] |
| Raw material cost | `material_cost_per_kg` | objective | Deterministic: solids fraction times blended resin price, plus solvent fraction times solvent price, plus catalyst. Prices are inputs |

The degree of cure is modeled with an Arrhenius-style rate times residence time (residence time is proportional to 1 / line speed), so **oven temperature and line speed interact**: a hot oven with a fast line can match a cooler oven with a slow line. That interaction is the main thing a good optimizer should exploit and OFAT would miss.

### Illustrative prices and run cost

| Item | Default | Note |
|---|---|---|
| PAI resin (solid basis) | $14.00/kg | Illustrative. PAI is generally more expensive than polyester-imide; exact prices are not public |
| PEI resin (solid basis) | $5.00/kg | Illustrative |
| Solvent blend | $1.80/kg | Illustrative |
| Catalyst | $40.00/kg | Illustrative |
| Fixed cost per line trial | $150 | Setup, technician time, testing. Illustrative |
| Energy/time cost | $3.00 x (oven_temp_c minus 400) x (30 / line_speed) | Hotter and slower runs cost more. Illustrative |
| Batch material in a trial | 40 kg of enamel | Illustrative |

All of these live in `config.yaml` so a user can replace them.

### Calibration rules for the simulator

- Fraction of the search box that is truly feasible: between 5 and 35 percent.
- The cheapest feasible formulation is **not** at a corner of the box on every variable (otherwise the problem is trivial).
- At least one spec is active at the cheapest feasible point (the optimum sits on a constraint boundary, as real optima usually do).
- Run cost varies by at least 3x across the box, so cost-awareness can matter.
- Noise is small relative to the spec margin but not zero (repeat runs differ visibly).

Once calibrated, **freeze** the simulator parameters before running the method comparison. Tuning the problem after seeing which method wins is benchmark cheating.

### Response model (illustrative)

The simulator in `src/formopt/problems/enamel.py` implements these equations. The coefficients shown are the starting values; the frozen, calibrated values are in `config.yaml` under `enamel.model`.

```
k        = A * exp(-Ea_over_R / (T + 273.15)) * (1 + c_cat * catalyst)     A = 5.4e4, Ea_over_R = 9000, c_cat = 1.5
dose     = k * (L_ref / line_speed)                                        L_ref = 60
cure     = 1 - exp(-dose)
over     = max(dose - dose_max, 0)                                         dose_max = 3.0
build    = b0 + b1 * (solids - 25)                                         b0 = 0.6, b1 = 0.02
breakdown= k_bd * cure * build * (1 + k_pai * pai) - k_over * over + N(0, sd_bd)
                                                                           k_bd = 40, k_pai = 0.25, k_over = 6, sd_bd = 1.0
flex_p   = logistic(f0 - f_over * over - f_sol * (solids - s_ref) / 7 + f_pai * pai)
           observed as Binomial(5, flex_p) / 5                             f0 = 3, f_over = 2.5, f_sol = 1.5, s_ref = 38, f_pai = 0.8
log_visc = log(v0) + v_sol * (solids - 25) + v_pai * pai + v_cat * catalyst + N(0, sd_logv)
                                                                           v0 = 800, v_sol = 0.09, v_pai = 0.9, v_cat = 0.1, sd_logv = 0.03
material_cost = solids/100 * (pai * p_pai + (1 - pai) * p_pei) + (1 - solids/100) * p_solv + catalyst/100 * p_cat
run_cost = fixed + batch_kg * material_cost + energy_rate * (T - 400) * (30 / line_speed)
                                                                           fixed = 150, batch_kg = 40, energy_rate = 3.0
```

## 4. The concrete emulator (real data)

The real-data benchmark uses the UCI Concrete Compressive Strength dataset [S2]: 1,030 rows, 8 inputs (seven ingredients in kg per m3 of concrete plus age in days, 1 to 365) and compressive strength in MPa. It is a true mixture/formulation dataset, which is why it was chosen.

### Ingredients and what they do

| Column | Role | Expert intuition |
|---|---|---|
| Cement | Main binder | More cement usually means more strength and more cost and CO2 |
| Blast furnace slag | Supplementary cementitious material (SCM) | Partial cement replacement, slower early strength gain |
| Fly ash | SCM | Partial cement replacement, cheap, slower early strength |
| Water | Reacts with cement, gives workability | Higher water-to-binder ratio means lower strength. This is the single strongest relationship in concrete |
| Superplasticizer | Admixture that improves flow at low water | Lets you cut water and keep workability |
| Coarse aggregate | Gravel or crushed stone | Bulk filler, cheap |
| Fine aggregate | Sand | Bulk filler, cheap |
| Age | Days of curing before the test | Strength rises with age. 28 days is the usual specification age |

USGS reports that fly ash and ground granulated blast furnace slag are increasingly used as partial substitutes for portland cement, and that blended cements were 63 percent of US shipments in the first nine months of 2025 [S3]. That is exactly the cost and sustainability trade-off this optimizer explores.

### Prices for the concrete problem

| Ingredient | Default price | Source |
|---|---|---|
| Cement | $160 per metric ton | USGS average mill value, 2025 estimate (portland, blended, and masonry combined) [S3] |
| Ground granulated blast furnace slag | $140 per metric ton | USGS: "about $140 or more for some GGBFS in 2025" [S4]. Upper-end, location dependent |
| Fly ash | $40 per metric ton | **ASSUMPTION.** No free authoritative price was found. Configurable |
| Water | $0.001 per kg | **ASSUMPTION.** Negligible compared with other ingredients |
| Superplasticizer | $2.00 per kg | **ASSUMPTION.** Admixture prices vary widely by product. Configurable |
| Coarse aggregate | $18.50 per metric ton | USGS crushed stone average unit value, 2025 estimate [S5] |
| Fine aggregate | $14.50 per metric ton | USGS construction sand and gravel average unit value, 2025 estimate [S6] |

These are national averages at the plant or quarry, not delivered prices. The conclusions are about method comparison, not about real concrete pricing.

Material cost per m3 is then `sum(kg_per_m3[i] * price_per_kg[i])`.

**Run cost (illustrative):** $250 fixed per trial batch (casting cylinders, curing, testing) plus the material cost of a 0.05 m3 trial batch. The 28-day wait is a calendar cost and is reported, not priced.

### Constraints for the concrete optimization

- Age is fixed at 28 days during optimization (the emulator is trained on all ages with age as an input).
- Each ingredient is bounded by its observed min and max in the data.
- Known constraints: total mass per m3 and water-to-binder ratio (water / (cement + slag + fly ash)) must stay inside their observed ranges in the data. These keep the optimizer from inventing "concrete" that has no physical meaning.
- Applicability domain: candidates must be within a distance threshold of real training mixes (see section 6). Outside it, the emulator is extrapolating and the result means nothing.
- Unknown constraint: 28-day strength at least the target (default 40 MPa, calibrate so 5 to 35 percent of the allowed region is feasible).

## 5. Common pitfalls (what makes output chemically or commercially naive)

1. **Treating raw material cost as unknown.** Recipe cost is arithmetic. Do not fit a GP to it. Model only what you have to measure.
2. **Learning known constraints.** Max oven temperature and mixture sums are checked directly, never modeled.
3. **Calling a noisy pass a pass.** In benchmarks, judge feasibility of the recommended formulation with the noise-free simulator or the emulator mean. Report both "lab said pass" and "truly passes".
4. **Ignoring the interaction between oven temperature and line speed.** Any analysis that varies them independently and reports "temperature does not matter" is wrong.
5. **Leakage in emulator validation.** The concrete data has the same mix tested at several ages, and some repeated rows. A random row split puts the same mix in train and test and inflates R squared. Use grouped cross-validation, grouped by the ingredient columns (the mix ID).
6. **Extrapolating the emulator.** The optimizer will happily push to the corners where the emulator is wrong and flattering. That is why the applicability domain check exists.
7. **Using the same model family for emulator and surrogate.** If the emulator is a GP with the same kernel as the optimizer's GP, BO looks unrealistically good. Use a different model class for the emulator.
8. **Unfair baselines.** Random search and DOE must get the same budget, the same seeds for the initial design where applicable, and the same cost accounting.
9. **Tuning on the test problem.** Calibrate the simulator, freeze it, then compare methods.
10. **Reporting one lucky run.** Always report many seeds with confidence intervals.

## 6. What a domain expert would immediately check

- Does breakdown go down when the film is undercured (low temperature, fast line)? It should.
- Does flexibility fail when overcured (very hot, very slow)? It should.
- Does viscosity rise with solids? It should, steeply.
- Is the recommended enamel inside the viscosity window with margin, not right on the edge?
- Is the recommended concrete mix's water-to-binder ratio reasonable, and is strength falling as that ratio rises in the emulator? Check with a partial dependence plot.
- Does the concrete emulator give 28-day strength that grows with age from 7 to 28 to 90 days for a fixed mix? It should.
- Is the "cheap" concrete mix cheap because it uses lots of fly ash? Plausible, but check it is inside the data's range of fly ash use, and note that early strength would be lower.
- Is the cost model per kg of enamel or per kg of cured film? This project uses per kg of liquid enamel as purchased. Say so.

## 7. Business framing

- **Cost per experiment** is the currency of R&D. Saving 30 percent of trials is real budget and real calendar time.
- **Time-to-spec** matters more than lab spend when a customer qualification window is at stake. Report experiments-to-target as well as dollars-to-target.
- **Raw material price swings** change the best recipe without changing the chemistry. Separating the property model from the cost model means re-optimizing on new prices costs zero new experiments.
- **Stage gates.** A director wants "expected spend to hit spec" so a project can be approved or killed early. The tool's estimate is a heuristic and must be labeled as one.
- **Operations link.** Fewer, better-chosen line trials free production capacity. This connects to operations management concepts like capacity utilization and the cost of quality.

## Sources

- [S1] US Patent 4,374,221, "High solids polyamide-imide magnet wire enamel." https://patents.google.com/patent/US4374221 (claims about 28 to 45 percent solids, viscosity 2,000 to 15,000 cP at 30 deg C, oven inlet about 260 to 371 deg C and outlet about 427 to 593 deg C, example wire speed 56 ft/min).
- [S2] Yeh, I-Cheng. Concrete Compressive Strength. UCI Machine Learning Repository, CC BY 4.0. https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength (DOI 10.24432/C5PK67). Original paper: Yeh, I-C. (1998), "Modeling of strength of high-performance concrete using artificial neural networks," Cement and Concrete Research 28(12).
- [S3] USGS Mineral Commodity Summaries 2026, Cement. https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-cement.pdf
- [S4] USGS Mineral Commodity Summaries 2026, Iron and Steel Slag. https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-iron-steel-slag.pdf
- [S5] USGS Mineral Commodity Summaries 2026, Stone (Crushed). https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-stone-crushed.pdf
- [S6] USGS Mineral Commodity Summaries 2026, Sand and Gravel (Construction). https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-sand-gravel.pdf
- Test methods for magnet wire (dielectric breakdown, elongation and mandrel wrap, heat shock, thermal class) are defined in ANSI/NEMA MW 1000 (overview: https://blog.ansi.org/ansi/ansi-nema-mw-01000-2025-magnet-wire-changes/) and the IEC 60851 series. An example class 200 dual-coat product standard is IEC 60317-29 (polyester or polyesterimide overcoated with polyamide-imide). These standards are paid documents; the project does not reproduce them.
