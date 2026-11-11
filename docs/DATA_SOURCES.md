# Data Sources

Every source here is free and public. Links and metadata were checked in November 2026. Raw data is never committed: `data/raw/` is gitignored, and `python -m formopt download` fetches everything. No accounts are needed.

Summary:

| # | Source | Role | Size | Account | License |
|---|---|---|---|---|---|
| 1 | UCI Concrete Compressive Strength | Primary real-data benchmark (emulator "lab") | 1,030 rows, about 34 KB zip | No | CC BY 4.0 |
| 2 | OpenML copy of the same concrete data | Fallback mirror for source 1 | Same | No | CC BY 4.0 |
| 3 | Enamel simulator | Domain demo | Generated in code | No | Synthetic, project's own |
| 4 | USGS Mineral Commodity Summaries 2026 | Ingredient price assumptions for concrete | PDFs, read by a human | No | US government work, public domain |
| - | Considered and not used | Olympus package and datasets, EDBO direct arylation | | | |

---

## 1. UCI Concrete Compressive Strength (primary real-data benchmark)

- **URL (landing page):** https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
- **Direct download:** https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip
- **DOI:** 10.24432/C5PK67. Creator: I-Cheng Yeh. Original paper: Yeh (1998), Cement and Concrete Research 28(12).
- **Contents:** `Concrete_Data.xls` (Excel 97 format, about 122 KB) and `Concrete_Readme.txt`. 1,030 rows, 9 columns: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate (all kg per m3), age (days, 1 to 365), and concrete compressive strength (MPa).
- **License:** Creative Commons Attribution 4.0 International.
- **Account needed:** No.
- **Format quirks and cleaning (`src/formopt/data.py`):**
  - The `.xls` file needs the `xlrd` package for pandas to read it (`pd.read_excel(path, engine="xlrd")`).
  - The column headers are long strings with units in parentheses, e.g. `"Cement (component 1)(kg in a m^3 mixture)"`. They are renamed to short snake_case names with a fixed mapping.
  - **Repeated mixes:** the same ingredient vector appears at several ages, and there are some exact duplicate rows. A `mix_id` hashes the seven ingredient columns (rounded to 0.1 kg) and is used for grouped cross-validation. Exact duplicate rows are dropped, and the count is logged.
  - Zeros are real values (no slag, no fly ash, no superplasticizer), not missing data.
  - Strength is right-skewed and spans roughly 2 to 83 MPa. The emulator is fit on strength directly.
- **How it is used:** an emulator (a tree ensemble chosen by grouped cross-validation) is trained on all rows with age as an input. During optimization, age is fixed at 28 days and the emulator mean plus Gaussian noise (sd equal to the emulator's cross-validated residual sd, capped) plays the role of the lab.

**Download steps (PowerShell):**

```powershell
# Preferred: the project command (downloads, unzips and checks the file)
uv run python -m formopt download

# Manual equivalent
New-Item -ItemType Directory -Force -Path data\raw\concrete | Out-Null
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip" -OutFile data\raw\concrete\concrete.zip
Expand-Archive -Path data\raw\concrete\concrete.zip -DestinationPath data\raw\concrete -Force
```

If the UCI site is down, the download falls back to source 2.

## 2. OpenML mirror (fallback for source 1)

- **URL:** https://www.openml.org/search?type=data&id=44959 (dataset `concrete_compressive_strength`, version 7, CC BY 4.0, default target `strength`).
- **Direct file:** https://openml.org/data/v1/download/22111823/concrete_compressive_strength.arff
- **How it is fetched:** `sklearn.datasets.fetch_openml(data_id=44959, as_frame=True)`.
- **Quirk:** column names differ from the UCI Excel file; both are mapped to the same short names in `data.py`.

## 3. Enamel simulator (synthetic, generated in code)

- Lives in `src/formopt/problems/enamel.py`. No download. Parameters live in `config.yaml` under `enamel:`.
- **Clearly synthetic.** It is a plausible illustrative model, not a calibrated model of any real enamel or of any company's products.
- Plausible ranges for solids (about 28 to 45 percent), viscosity window (2,000 to 15,000 cP at 30 deg C), and oven zone temperatures (outlet about 427 to 593 deg C) come from US Patent 4,374,221 (https://patents.google.com/patent/US4374221). Everything else (resin prices, catalyst effect size, run costs, the exact response equations) is illustrative.

## 4. USGS Mineral Commodity Summaries 2026 (price assumptions)

Read by a human, not downloaded by the code. Values are copied into `config.yaml` with a comment citing the PDF.

| Item | Value used | URL |
|---|---|---|
| Cement average mill value, 2025 estimate | $160/t | https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-cement.pdf |
| GGBFS price ("about $140 or more for some GGBFS in 2025") | $140/t | https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-iron-steel-slag.pdf |
| Crushed stone average unit value, 2025 estimate | $18.50/t | https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-stone-crushed.pdf |
| Construction sand and gravel average unit value, 2025 estimate | $14.50/t | https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-sand-gravel.pdf |
| Fly ash | $40/t | **ASSUMPTION**, no free authoritative source found. Configurable in `config.yaml` |
| Superplasticizer | $2.00/kg | **ASSUMPTION**, same note |

## Considered and not used (and why)

| Candidate | Finding | Decision |
|---|---|---|
| `olymp` package on PyPI (Olympus) | Latest release is 0.0.1b0 from October 2020. Tested on Python 3.12: `uv add olymp` installs, but `import olympus` fails without scikit-learn and then fails with `AttributeError: module 'matplotlib.pyplot' has no attribute 'register_cmap'` on current matplotlib. Its pre-trained emulators need old TensorFlow (`tensorflow==1.15` in its extras), which does not exist for Python 3.12 | Not used |
| Olympus P3HT/CNT conductivity data (Bash et al., 2021) | 178 rows, a real five-component polymer mixture, MIT licensed | Not used; a possible second real-data benchmark |
| EDBO direct arylation dataset (Shields et al., Nature 590, 89 to 96, 2021) | Available at https://github.com/b-shields/edbo (MIT), file `experiments/data/direct_arylation/experiment_index.csv`, 1,728 reactions, about 170 KB. Variables are base, ligand, solvent (categorical SMILES), concentration and temperature (each only 2 or 3 levels) | Not used: mostly categorical, a lookup table not a continuous formulation |
| Olympus crossed barrel (Gongora et al., 2020) | 600 rows, 4 continuous variables, mechanical toughness of 3D printed structures | Real and clean, but not a formulation. Not used |
| Olympus SnAr, AgNP, HPLC | Real chemistry data, small and process-oriented | Not used |

## If the download is blocked

If `archive.ics.uci.edu` and `openml.org` are both blocked on your network, download the UCI zip in a browser from the landing page above and put it at `data\raw\concrete\concrete.zip`.
