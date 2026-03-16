# Action Plan: Dublin Airbnb ML Notebook → Dash Dashboard
## Rubric-Aligned Structure

---

## Rubric Mapping

| Tab # | Dashboard Tab Title            | Notebook Sections | Rubric Dimension                    | Points |
|-------|-------------------------------|-------------------|-------------------------------------|--------|
| 1     | Data Loading & Wrangling       | 2 + 3             | 1. Data Acquisition & Wrangling     | 20     |
| 2     | EDA                            | 4–7 (SQL + EDA)   | 2. EDA                              | 20     |
| 3     | Machine Learning               | 8–9               | 3. Machine Learning Modelling       | 25     |
| 4     | Interactive App / Dashboard    | 10                | 4. Interactive App / Dashboard      | 15     |
| 5     | Storytelling & Communication   | 11                | 5. Storytelling & Communication     | 10     |
| 6     | Critical Thinking & Reflection | 12                | 6. Critical Thinking & Reflection   | 10     |

> **Note:** Tabs 5 and 6 are narrative/text tabs. They do not contain charts but are required by the rubric and should be visible, well-formatted sections of the dashboard.

---

## Tab 1 — Data Loading & Wrangling
**Rubric Dimension:** 1. Data Acquisition & Wrangling (20 pts)
**Purpose:** Demonstrate that data was correctly sourced, inspected, and cleaned. Corresponds to notebook Sections 2 (Data Acquisition) and 3 (Data Cleaning & Feature Engineering).

### Content

**KPI summary strip (top of tab):**
- Total listings loaded
- Total reviews loaded
- Number of neighbourhoods
- Number of features after cleaning
- Count of engineered features

**Missing Value Audit (Section 3.1):**
- Horizontal Plotly bar chart: top 25 columns by % missing
- Interactive: slider to control how many columns are shown (5–25)

**Price Sanity Check (Section 3.2 / cell 13):**
- `px.histogram` — raw price distribution with median line annotation
- `px.histogram` — log(price+1) distribution
- Displayed side-by-side via `make_subplots`

**Room Type Overview (Section 3.2 / cell 14):**
- `px.bar` — listing count by room type
- `px.box` — price by room type (outliers hidden)
- Side-by-side

**Neighbourhood Summary Table (cell 15):**
- `dash_table.DataTable` — sortable, filterable
- Columns: neighbourhood, n_listings, median_price, avg_rating, pct_superhost, avg_host_tenure

**Correlation Matrix (cells 16 / 39):**
- `px.imshow` — annotated heatmap, RdYlGn colour scale
- Hover tooltip shows exact r value and feature pair

---

## Tab 2 — EDA (Pricing, Sentiment, Geography)
**Rubric Dimension:** 2. EDA (20 pts)
**Purpose:** Answer the core business question ("What drives nightly prices in Dublin?") through visual exploration. Covers notebook Sections 4 (SQL Aggregations) and 5 (EDA).

### SQL Aggregation Charts (Section 4)
Each query is shown as a labelled chart + expandable DataTable. Grouped in a sub-section titled "SQL Aggregations".

| Query | Chart type | Interactive element |
|-------|-----------|---------------------|
| Q1 — Neighbourhood price summary | Horizontal `px.bar`, sorted by avg price | None |
| Q2 — Price by room type × neighbourhood | `px.bar` grouped | Neighbourhood multi-select dropdown |
| Q3 — Superhost price premium by room type | `px.bar` grouped | None |
| Q4 — Price vs guest capacity | `px.scatter` + trend line | None |
| Q5 — Availability tiers vs price | `px.bar` | None |
| Q6 — Multi-listing vs single-listing hosts | `px.bar` | None |
| Q7 — Price tier vs review scores | `px.imshow` heatmap | None |
| Q8 — Minimum-night policy distribution | `px.bar` | None |

### EDA Charts (Section 5)

**5.1 Missingness heatmap** → already shown in Tab 1; cross-reference note displayed here.

**5.2 Correlation matrix** → already shown in Tab 1; cross-reference note displayed here.

**5.3 Room-type breakdown (cell 42):**
- `px.bar` — listing count by room type
- `px.box` — price by room type
- Neighbourhood dropdown filters both charts

**5.4 Monthly review volume / demand proxy (cell 45):**
- `px.line` with filled area — monthly reviews from 2018 onward
- COVID-19 restriction band annotated with `add_vrect`
- Date-range slider (RangeSlider) for zooming

**5.5 Superhost vs non-superhost (cell 48):**
- `px.box` — price by superhost status
- `px.box` — review score by superhost status
- Side-by-side; summary median table below

**5.6 Minimum nights distribution (cell 51):**
- `px.histogram` — capped distribution
- `px.bar` — policy bucket counts (1–2 / 3–7 / 8–29 / 30+)
- Slider to set cap on histogram x-axis (max 60 nights)

**5.7 Host portfolio size (cell 54):**
- `px.histogram` — listing count per host (capped)
- `px.bar` — host type buckets
- Slider to set cap on histogram x-axis (max 20 listings)

---

## Tab 3 — Machine Learning Modelling
**Rubric Dimension:** 3. Machine Learning Modelling (25 pts)
**Purpose:** Show the full ML workflow: feature prep, cross-validation, hyperparameter tuning, final hold-out evaluation, and feature importance. Corresponds to notebook Sections 8–9 (ML pipeline through feature importance + SHAP).

### Sub-section A — Model Cross-Validation (Section 4B / cell 73)
- `px.bar` horizontal — CV RMSE per model with error bars
- `px.bar` horizontal — CV R² per model with error bars
- Radio-button metric toggle: RMSE / MAE / R² (single chart updates)
- Sortable `DataTable` of `cv_results_df`

### Sub-section B — Hyperparameter Tuning (Section 5D / cells 77–78)
- `px.bar` horizontal — Best post-tune CV RMSE per model (viz 1)
- `px.bar` grouped — Pre-tune vs post-tune RMSE side-by-side (viz 2)
- `px.bar` horizontal — RMSE improvement from tuning (viz 3)
- `px.imshow` — ranking heatmap: RMSE rank × improvement rank (viz 4)
- Dropdown: select a model → show its top-15 hyperparameter config trajectory (`px.bar` horizontal, viz 5)

### Sub-section C — Final Hold-Out Test Results (cells 80–83)
- `dash_table.DataTable` — all models × all metrics (RMSE, MAE, R², MAPE, Median AE in log space and € space)
- `dash_table.DataTable` — best model per metric summary
- Highlighted banner showing the winning model name + its RMSE/MAE/R²

### Sub-section D — Feature Importance (cell 85)
- `px.bar` horizontal — top-N features by importance, coloured by importance share
- Dropdown: top-10 / top-20 / all features
- Model name and metric type (feature_importances_ vs abs. coef) shown as subtitle

### Sub-section E — SHAP (if computed)
- If `shap` values are available in the notebook's final output, a beeswarm or summary bar will be added here using `px.scatter` (SHAP does not integrate directly with Plotly; values will be extracted and plotted manually)
- If SHAP is not computed in the notebook, this sub-section is omitted

---

## Tab 4 — Interactive App / Dashboard
**Rubric Dimension:** 4. Interactive App / Dashboard (15 pts)
**Purpose:** This tab *is* the demonstration of interactivity itself. It contains a self-contained interactive price explorer that lets a user query the cleaned dataset without scrolling through other tabs.

### Price Explorer Panel
- Filters (sidebar or top bar):
  - Neighbourhood multi-select dropdown
  - Room type checklist
  - Price range slider (€0–€1,000+)
  - Accommodates range slider (1–16)
  - Superhost toggle (All / Superhost only / Non-superhost only)
- Outputs that update live based on filters:
  - KPI strip: filtered listing count, median price, avg rating
  - `px.histogram` — price distribution of filtered listings
  - `px.scatter` — price vs accommodates, coloured by room type
  - `px.box` — price by neighbourhood (top 15 by count), filtered
  - `dash_table.DataTable` — filtered listing subset (paginated, 20 rows/page)

> This tab serves double duty: it satisfies the rubric's interactivity dimension and gives a user-facing tool beyond the analytical charts on other tabs.

---

## Tab 5 — Storytelling & Communication
**Rubric Dimension:** 5. Storytelling & Communication (10 pts)
**Purpose:** Narrative summary of findings for a non-technical audience. No new computation — text and key callout numbers drawn from already-computed DataFrames.

### Content
- **Executive Summary** (3–4 paragraphs): what the data shows about Dublin Airbnb pricing
- **Key Findings callout cards** (styled dcc.Markdown or html.Div blocks):
  - Top price drivers (from feature importance)
  - Superhost price/rating finding
  - Seasonal demand pattern
  - Best-performing model + its test RMSE in €
- **Methodology note:** brief description of the data pipeline and modelling approach
- All numbers in this tab are dynamically pulled from computed DataFrames (no hardcoding)

---

## Tab 6 — Critical Thinking & Reflection
**Rubric Dimension:** 6. Critical Thinking & Reflection (10 pts)
**Purpose:** Honest assessment of limitations, risks, and possible extensions. Text-only tab.

### Content
- **Data limitations:** Inside Airbnb data is a snapshot; no booking/revenue ground truth
- **Model limitations:** log-price target, potential target leakage from review scores, no spatial autocorrelation modelling
- **Ethical considerations:** pricing inequality across neighbourhoods, impact on local housing
- **Future work:** time-series forecasting, map-based choropleth, live calendar data integration, SHAP interaction plots
- **Reflection on process:** what changed between initial design and final implementation

---

## Interactive Elements — Full List

| Element | Tab | Reason |
|---------|-----|--------|
| Bin-count slider | Tab 1 (price histogram) | Price histograms are sensitive to bin choice |
| Top-N slider | Tab 1 (missing values) | User may want to inspect fewer or more columns |
| Neighbourhood multi-select dropdown | Tab 2 (SQL Q2, EDA 5.3) | Core analytical dimension |
| Date-range RangeSlider | Tab 2 (monthly reviews) | Zooming into post-COVID recovery is a key insight |
| Cap slider (min nights) | Tab 2 (EDA 5.6) | Long-tail distributions need capping to be readable |
| Cap slider (host portfolio) | Tab 2 (EDA 5.7) | Same reason |
| Metric toggle (radio buttons) | Tab 3 (CV comparison) | Saves vertical space vs three separate charts |
| Model selector dropdown | Tab 3 (tuning trajectory) | One chart per model avoids clutter |
| Top-N feature slider | Tab 3 (feature importance) | Allows exploration beyond top-20 |
| Price / accommodates / room type / superhost / neighbourhood filters | Tab 4 (Price Explorer) | Core interactivity showcase for rubric dimension 4 |

---

## File / Component Structure

```
dublin_airbnb_dashboard/
│
├── app.py                    # Entry point: Dash app init, layout assembly, server
│
├── data_loader.py            # Sections 2–3: remote data fetch + full cleaning pipeline
│                             # Exposes: df, reviews_raw, nbhd_summary, corr_matrix, etc.
│
├── sql_queries.py            # Section 4: all 8 pandasql queries
│                             # Exposes: q1 through q8 as DataFrames
│
├── ml_pipeline.py            # Sections 8–9: feature prep, CV, tuning, eval, importance
│                             # Exposes: cv_results_df, tuning_df, tuning_comparison_df,
│                             #          comparison_df, best_model_per_metric_df,
│                             #          importance_df, grid_objects, winning_model_name,
│                             #          all_results
│
├── figures.py                # All Plotly figure factory functions (one per chart)
│                             # Pure functions: take DataFrames → return go.Figure
│
├── layout/
│   ├── tab_wrangling.py      # Tab 1 layout
│   ├── tab_eda.py            # Tab 2 layout
│   ├── tab_ml.py             # Tab 3 layout
│   ├── tab_interactive.py    # Tab 4 layout (Price Explorer)
│   ├── tab_storytelling.py   # Tab 5 layout (text/narrative)
│   └── tab_reflection.py     # Tab 6 layout (text/reflection)
│
├── callbacks/
│   ├── cb_wrangling.py       # Callbacks for Tab 1 sliders
│   ├── cb_eda.py             # Callbacks for Tab 2 dropdowns, sliders, date range
│   ├── cb_ml.py              # Callbacks for Tab 3 metric toggle, model dropdown
│   └── cb_interactive.py     # Callbacks for Tab 4 Price Explorer filters
│
├── assets/
│   └── style.css             # Airbnb brand palette, card styles, typography
│
└── requirements.txt
```

`app.py` runs `data_loader`, `sql_queries`, and `ml_pipeline` **once at startup**. All results are stored as module-level variables. Callbacks only filter/reshape already-computed data — no re-running of models inside callbacks.

---

## Dependencies

### Preserved from notebook (no changes)
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`, `lightgbm`
- `shap` (used if SHAP output exists in notebook)
- `pandasql`

### New — required for dashboard
- `dash >= 2.16`
- `dash-bootstrap-components` (tab/card/grid layout)
- `plotly >= 5.20`

### Removed (notebook-only)
- `matplotlib`
- `seaborn`

### Data — all loaded remotely at startup

| File | URL |
|------|-----|
| `listings.csv.gz` | `https://github.com/yanazzz315-cloud/Dublin-Listing/raw/main/listings.csv.gz` |
| `reviews.csv` | `.../reviews.csv` |
| `neighbourhoods.csv` | `.../neighbourhoods.csv` |
| `neighbourhoods.geojson` | `.../neighbourhoods.geojson` |
| `calendar.csv.gz` | `.../calendar.csv.gz` |
| `listings.csv` | `.../listings.csv` |

No local data files are required.

---

## Risks & Assumptions

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **ML runtime:** GridSearchCV across 9 models takes 10–30+ min | High | Cache fitted pipelines and result DataFrames to disk with `joblib`; reload on subsequent starts |
| **GitHub data availability:** all data is fetched live from a personal repo | Medium | Document clearly; add try/except with a user-friendly error message if fetch fails |
| **`pandasql` performance:** 8 queries on a large DataFrame at startup | Low | Run once at startup and store results; never re-run inside callbacks |
| **`dwelling_type` column:** referenced in cell 60 but not in standard Inside Airbnb schema | Medium | Already partially guarded in cell 63 (`if col in df_ml.columns`); ensure full guard in `data_loader.py` |
| **`all_results` dict:** populated in cells between 80–82 — full definition not visible in excerpts | Medium | Confirm exact cell before coding `ml_pipeline.py` |
| **SHAP:** imported in cell 2 but no SHAP plot produced in notebook | Low | Omit from dashboard unless a SHAP cell is found; placeholder sub-section added |

### Assumptions

1. Dashboard runs locally; no auth or multi-user handling needed.
2. `SEED = 123` and `random_state = 42` preserved unchanged.
3. Airbnb brand palette `['#FF5A5F', '#00A699', '#FC642D', '#484848', '#767676', '#FFB400']` used throughout.
4. `log_price` target retained; metrics shown in both log space and € space.
5. No choropleth map planned (GeoJSON loaded but never plotted in notebook); can be added to Tab 4 as a future enhancement.
6. All `display()` (IPython) calls replaced with `dash_table.DataTable`.
7. Tabs 5 and 6 contain authored narrative text — these will need to be written as part of the implementation, drawing numbers dynamically from computed DataFrames.

---

*Plan updated to align with rubric. Awaiting approval before any code is written.*
