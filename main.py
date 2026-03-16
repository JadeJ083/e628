# =============================================================================
# Dublin Airbnb — Interactive Dash Dashboard
# Drop-in replacement for the starter app.py deployed on Render.
# All analytical logic is preserved verbatim from the notebook.
# Only the rendering layer is changed: matplotlib/seaborn → Plotly.
# =============================================================================

import warnings, os, io, gzip
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandasql import sqldf

from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# ── Brand palette ─────────────────────────────────────────────────────────────
AIRBNB_PALETTE = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676", "#FFB400"]
AIRBNB_RED     = "#FF5A5F"
AIRBNB_TEAL    = "#00A699"
SEED           = 123
CITY_NAME      = "Dublin"

# =============================================================================
# SECTION 1 — DATA ACQUISITION  (notebook cells 6–7)
# =============================================================================
GITHUB_BASE         = "https://github.com/yanazzz315-cloud/Dublin-Listing/raw/main/"
LISTINGS_GZ_URL     = GITHUB_BASE + "listings.csv.gz"
REVIEWS_CSV_URL     = GITHUB_BASE + "reviews.csv"
NEIGHBOURHOODS_URL  = GITHUB_BASE + "neighbourhoods.csv"

print("Loading data from GitHub …")
try:
    listings_raw = pd.read_csv(LISTINGS_GZ_URL, low_memory=False)
    reviews_raw  = pd.read_csv(REVIEWS_CSV_URL,  low_memory=False)
    print(f"  Listings : {len(listings_raw):,} rows × {listings_raw.shape[1]} cols")
    print(f"  Reviews  : {len(reviews_raw):,} rows")
except Exception as e:
    raise RuntimeError(f"Data load failed. Check GitHub URLs.\n{e}")

# =============================================================================
# SECTION 2 — DATA CLEANING & FEATURE ENGINEERING  (notebook cells 9–11)
# =============================================================================
def parse_price(df):
    return df.assign(
        price=lambda x: (
            x["price"].astype(str)
            .str.replace(r"[\$\u20ac,]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
    )

def parse_numeric_cols(df):
    cols = [
        "review_scores_rating","review_scores_cleanliness",
        "review_scores_communication","review_scores_location",
        "review_scores_value","accommodates","bedrooms","beds",
        "minimum_nights","number_of_reviews",
        "calculated_host_listings_count","latitude","longitude"
    ]
    existing = [c for c in cols if c in df.columns]
    return df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in existing})

def add_derived_features(df):
    return df.assign(
        price_per_person=lambda x: x["price"] / x["accommodates"].replace(0, np.nan),
        host_age_years=lambda x: (
            (pd.Timestamp.now() - pd.to_datetime(x.get("host_since"), errors="coerce"))
            .dt.days / 365
        ),
        host_identity_verified=lambda x: (
            x.get("host_identity_verified", pd.Series("f", index=x.index))
             .map({"t": 1, "f": 0})
        ),
        is_superhost=lambda x: (
            x.get("host_is_superhost", pd.Series("f", index=x.index))
             .map({"t": 1, "f": 0})
        ),
        log_price=lambda x: np.log1p(x["price"])
    )

def handle_missing_values(df):
    return df.dropna(subset=["latitude", "longitude", "price"])

def filter_price_outliers(df, lo=15, hi=1000):
    return df.query("@lo <= price <= @hi")

print("Cleaning data …")
df = (
    listings_raw
    .pipe(parse_price)
    .pipe(parse_numeric_cols)
    .pipe(add_derived_features)
    .pipe(handle_missing_values)
    .pipe(filter_price_outliers)
    .reset_index(drop=True)
)
print(f"  Clean dataset: {len(df):,} listings")

# Missing-value report
missing_report = (
    df.isnull().sum()
    .pipe(lambda s: s[s > 0])
    .sort_values(ascending=False)
    .rename("n_missing")
    .to_frame()
    .assign(pct=lambda x: (x["n_missing"] / len(df) * 100).round(1))
    .head(25)
    .reset_index()
    .rename(columns={"index": "column"})
)

# Neighbourhood summary
nbhd_summary = (
    df.groupby("neighbourhood_cleansed", as_index=False)
    .agg(
        listings       =("price", "count"),
        median_price   =("price", "median"),
        avg_rating     =("review_scores_rating", "mean"),
        pct_superhost  =("is_superhost", "mean"),
        avg_host_tenure=("host_age_years", "mean"),
    )
    .sort_values("median_price", ascending=False)
    .round(2)
)

# Correlation matrix
corr_cols = [
    "price","accommodates","bedrooms","beds","minimum_nights",
    "number_of_reviews","review_scores_rating","review_scores_cleanliness",
    "review_scores_location","review_scores_value",
    "host_age_years","is_superhost","calculated_host_listings_count"
]
corr_cols   = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].dropna().corr()

# =============================================================================
# SECTION 3 — SQL AGGREGATIONS  (notebook cells 20–34)
# =============================================================================
print("Running SQL aggregations …")
sql = lambda q: sqldf(q, {"df": df})

q1 = sql("""
    SELECT neighbourhood_cleansed AS neighbourhood,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(MIN(price),2) AS min_price,
           ROUND(MAX(price),2) AS max_price,
           ROUND(AVG(price_per_person),2) AS avg_price_per_person
    FROM df GROUP BY neighbourhood_cleansed
    HAVING n_listings >= 10 ORDER BY avg_price DESC
""")

q2 = sql("""
    SELECT neighbourhood_cleansed AS neighbourhood, room_type,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(review_scores_rating),2) AS avg_rating
    FROM df GROUP BY neighbourhood_cleansed, room_type
    HAVING n_listings >= 5 ORDER BY neighbourhood, avg_price DESC
""")

q3 = sql("""
    SELECT room_type,
           CASE WHEN is_superhost=1 THEN 'Superhost' ELSE 'Non-superhost' END AS host_status,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(review_scores_rating),2) AS avg_rating,
           ROUND(AVG(number_of_reviews),1) AS avg_reviews
    FROM df GROUP BY room_type, host_status ORDER BY room_type, host_status
""")

q4 = sql("""
    SELECT accommodates, COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(bedrooms),1) AS avg_bedrooms,
           ROUND(AVG(beds),1) AS avg_beds
    FROM df WHERE accommodates <= 10
    GROUP BY accommodates ORDER BY accommodates
""")

q5 = sql("""
    SELECT CASE
               WHEN availability_365=0 THEN 'Unavailable'
               WHEN availability_365 BETWEEN 1 AND 90 THEN 'Low (1-90 days)'
               WHEN availability_365 BETWEEN 91 AND 270 THEN 'Medium (91-270)'
               ELSE 'High (271-365 days)'
           END AS availability_tier,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(minimum_nights),1) AS avg_min_nights,
           ROUND(AVG(number_of_reviews),1) AS avg_reviews
    FROM df GROUP BY availability_tier ORDER BY avg_price DESC
""")

q6 = sql("""
    SELECT CASE
               WHEN calculated_host_listings_count=1 THEN '1 listing'
               WHEN calculated_host_listings_count BETWEEN 2 AND 5 THEN '2-5 listings'
               ELSE '6+ listings'
           END AS host_portfolio,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(review_scores_rating),2) AS avg_rating,
           ROUND(AVG(is_superhost)*100,1) AS pct_superhost
    FROM df GROUP BY host_portfolio ORDER BY n_listings DESC
""")

q7 = sql("""
    SELECT CASE
               WHEN price<80  THEN '1. Budget (<80)'
               WHEN price<150 THEN '2. Mid (80-149)'
               WHEN price<300 THEN '3. Upper (150-299)'
               ELSE '4. Premium (300+)'
           END AS price_tier,
           COUNT(*) AS n_listings,
           ROUND(AVG(review_scores_rating),3) AS avg_overall,
           ROUND(AVG(review_scores_cleanliness),3) AS avg_clean,
           ROUND(AVG(review_scores_location),3) AS avg_location,
           ROUND(AVG(review_scores_value),3) AS avg_value
    FROM df WHERE review_scores_rating IS NOT NULL
    GROUP BY price_tier ORDER BY price_tier
""")

q8 = sql("""
    SELECT CASE
               WHEN minimum_nights=1 THEN '1 night'
               WHEN minimum_nights BETWEEN 2 AND 3 THEN '2-3 nights'
               WHEN minimum_nights BETWEEN 4 AND 7 THEN '4-7 nights'
               WHEN minimum_nights BETWEEN 8 AND 30 THEN '8-30 nights'
               ELSE '30+ nights'
           END AS min_night_bucket,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM df),1) AS pct_of_total
    FROM df GROUP BY min_night_bucket ORDER BY n_listings DESC
""")

# Monthly review time-series
review_ts = (
    reviews_raw
    .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
    .dropna(subset=["date"])
    .set_index("date")
    .resample("ME").size()
    .rename("review_count")
    .reset_index()
    .query('date >= "2018-01-01"')
)

# Superhost comparison
superhost_comp = (
    df
    .assign(superhost_label=lambda x: x["is_superhost"].map({1:"Superhost",0:"Non-superhost"}))
    .dropna(subset=["is_superhost","price","review_scores_rating"])
)

# =============================================================================
# SECTION 4 — MACHINE LEARNING  (notebook cells 57–85)
# Cache results to disk so GridSearch only runs once.
# =============================================================================
CACHE_FILE = "ml_cache.joblib"

def run_ml_pipeline():
    print("Running ML pipeline (this may take several minutes) …")
    df_ml = df.copy()

    if "log_price" not in df_ml.columns:
        df_ml["log_price"] = np.log1p(df_ml["price"])

    for col in ["host_response_rate","host_acceptance_rate"]:
        if col in df_ml.columns:
            df_ml[col] = (
                df_ml[col].astype(str).str.replace("%","",regex=False)
                .replace("nan", np.nan)
            )
            df_ml[col] = pd.to_numeric(df_ml[col], errors="coerce")

    if "amenities" in df_ml.columns:
        df_ml["amenity_count"] = (
            df_ml["amenities"].fillna("")
            .apply(lambda x: len([a for a in str(x).strip("{}").split(",") if a.strip()]))
        )

    if "bathrooms" in df_ml.columns:
        df_ml["bathrooms"] = pd.to_numeric(df_ml["bathrooms"], errors="coerce")
    if ("bathrooms" not in df_ml.columns) or (df_ml["bathrooms"].isna().all()):
        if "bathrooms_text" in df_ml.columns:
            df_ml["bathrooms"] = (
                df_ml["bathrooms_text"].astype(str)
                .str.extract(r"(\d+\.?\d*)")[0].astype(float)
            )

    binary_map = {"t":1,"f":0,True:1,False:0}
    for col in ["host_is_superhost","host_identity_verified","host_has_profile_pic"]:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(binary_map).fillna(df_ml[col])

    df_ml["dwelling_type"] = (
        df_ml["property_type"].str.lower()
        .str.replace(
            r"^(entire|room in|private room in|private room|shared room in|hotel room in)\s*",
            "", regex=True
        ).str.strip()
    )

    from math import radians, cos, sin, asin, sqrt
    DUBLIN_CENTER_LAT, DUBLIN_CENTER_LON = 53.3498, -6.2603
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2-lat1, lon2-lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return R * 2 * asin(sqrt(a))
    df_ml["dist_to_center_km"] = df_ml.apply(
        lambda r: haversine_km(r["latitude"],r["longitude"],DUBLIN_CENTER_LAT,DUBLIN_CENTER_LON), axis=1
    )

    feature_cols_raw = [
        "room_type","dwelling_type","neighbourhood_cleansed",
        "accommodates","bathrooms","bedrooms","beds",
        "host_response_rate","host_acceptance_rate",
        "review_scores_rating","review_scores_cleanliness",
        "review_scores_communication","review_scores_location","review_scores_value",
        "number_of_reviews","calculated_host_listings_count",
        "host_age_years","is_superhost","host_identity_verified",
        "amenity_count","longitude","latitude","dist_to_center_km",
        "minimum_nights"
    ]
    feature_cols = [c for c in feature_cols_raw if c in df_ml.columns]
    model_df = df_ml[feature_cols + ["price","log_price"]].copy().dropna(subset=["log_price"]).reset_index(drop=True)

    X = model_df.drop(columns=["price","log_price"])
    y = model_df["log_price"]

    numeric_features     = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    num_pipe  = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    num_scaled= Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    cat_pipe  = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        [("num",num_pipe,numeric_features),("cat",cat_pipe,categorical_features)], remainder="drop"
    )
    preprocessor_scaled = ColumnTransformer(
        [("num",num_scaled,numeric_features),("cat",cat_pipe,categorical_features)], remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # ── Cross-validation ──────────────────────────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Lasso":             Lasso(alpha=0.001, max_iter=10000),
        "Decision Tree":     DecisionTreeRegressor(random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN":               KNeighborsRegressor(n_neighbors=5),
        "XGBoost":           XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=42, objective="reg:squarederror", n_jobs=-1),
        "LightGBM":          LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                            max_depth=-1, random_state=42, verbose=-1),
    }
    linear_model_names = {"Linear Regression","Ridge","Lasso"}
    pipelines = {
        name: Pipeline([("preprocessor", preprocessor_scaled if name in linear_model_names else preprocessor),
                        ("model", mdl)])
        for name, mdl in models.items()
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"rmse":"neg_root_mean_squared_error","mae":"neg_mean_absolute_error","r2":"r2"}
    cv_results = []
    for name, pipe in pipelines.items():
        print(f"  CV: {name}")
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                                return_train_score=False, n_jobs=-1)
        cv_results.append({
            "Model": name,
            "CV RMSE Mean": -scores["test_rmse"].mean(),
            "CV RMSE Std":   scores["test_rmse"].std(),
            "CV MAE Mean":  -scores["test_mae"].mean(),
            "CV MAE Std":    scores["test_mae"].std(),
            "CV R2 Mean":    scores["test_r2"].mean(),
            "CV R2 Std":     scores["test_r2"].std(),
        })

    cv_results_df = (
        pd.DataFrame(cv_results)
        .sort_values("CV RMSE Mean").reset_index(drop=True)
    )
    top_3_models = cv_results_df["Model"].head(3).tolist()
    print(f"  Top 3 models: {top_3_models}")

    # ── Hyperparameter tuning (top 3 only) ────────────────────────────────────
    tuning_configs = {
        "Linear Regression": {"model": LinearRegression(),           "params": {}},
        "Ridge":             {"model": Ridge(),                       "params": {"model__alpha":[0.01,0.1,1,10,100]}},
        "Lasso":             {"model": Lasso(max_iter=10000),         "params": {"model__alpha":[0.001,0.01,0.1,1]}},
        "Decision Tree":     {"model": DecisionTreeRegressor(random_state=42),
                              "params": {"model__max_depth":[3,5,8,None],"model__min_samples_split":[2,5,10]}},
        "Random Forest":     {"model": RandomForestRegressor(random_state=42,n_jobs=-1),
                              "params": {"model__n_estimators":[100,200],"model__max_depth":[10,20,None]}},
        "Gradient Boosting": {"model": GradientBoostingRegressor(random_state=42),
                              "params": {"model__n_estimators":[100,200],"model__learning_rate":[0.05,0.1],"model__max_depth":[3,4]}},
        "KNN":               {"model": KNeighborsRegressor(),
                              "params": {"model__n_neighbors":[3,5,7,11],"model__weights":["uniform","distance"]}},
        "XGBoost":           {"model": XGBRegressor(objective="reg:squarederror",random_state=42,n_jobs=-1),
                              "params": {"model__n_estimators":[100,200],"model__learning_rate":[0.05,0.1],"model__max_depth":[3,5]}},
        "LightGBM":          {"model": LGBMRegressor(random_state=42,verbose=-1),
                              "params": {"model__n_estimators":[100,200],"model__learning_rate":[0.05,0.1],"model__num_leaves":[31,50]}},
    }
    tuning_cv     = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_objects  = {}
    tuning_rows   = []
    for name in top_3_models:
        if name not in tuning_configs:
            continue
        cfg = tuning_configs[name]
        prep = preprocessor_scaled if name in linear_model_names else preprocessor
        pipe = Pipeline([("preprocessor",prep),("model",cfg["model"])])
        grid = GridSearchCV(pipe, cfg["params"], cv=tuning_cv,
                            scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True)
        print(f"  Tuning: {name}")
        grid.fit(X_train, y_train)
        grid_objects[name] = grid
        tuning_rows.append({"Model":name,"Best CV RMSE":-grid.best_score_,"Best Parameters":grid.best_params_})

    tuning_df = pd.DataFrame(tuning_rows).sort_values("Best CV RMSE").reset_index(drop=True)
    tuning_df["Rank"] = range(1, len(tuning_df)+1)

    pretune_lookup  = cv_results_df.set_index("Model")
    posttune_lookup = tuning_df.set_index("Model")
    comparison_rows = []
    for name in tuning_df["Model"]:
        if name in pretune_lookup.index:
            pre  = pretune_lookup.loc[name,"CV RMSE Mean"]
            post = posttune_lookup.loc[name,"Best CV RMSE"]
            comparison_rows.append({"Model":name,"Pre-tune CV RMSE":pre,"Post-tune CV RMSE":post,"Improvement":pre-post})
    tuning_comparison_df = pd.DataFrame(comparison_rows).sort_values("Post-tune CV RMSE").reset_index(drop=True)

    # ── Final hold-out evaluation ─────────────────────────────────────────────
    y_test_price = np.expm1(y_test)
    all_results  = {}
    for name in tuning_df["Model"]:
        pipe        = grid_objects[name].best_estimator_
        y_pred_log  = pipe.predict(X_test)
        y_pred_price= np.expm1(y_pred_log)
        all_results[name] = {
            "pipeline":     pipe,
            "y_pred_log":   y_pred_log,
            "y_pred_price": y_pred_price,
            "metrics": {
                "RMSE (log)":    np.sqrt(mean_squared_error(y_test, y_pred_log)),
                "MAE (log)":     mean_absolute_error(y_test, y_pred_log),
                "R² (log)":      r2_score(y_test, y_pred_log),
                "RMSE (€)":      np.sqrt(mean_squared_error(y_test_price, y_pred_price)),
                "MAE (€)":       mean_absolute_error(y_test_price, y_pred_price),
                "R² (€)":        r2_score(y_test_price, y_pred_price),
                "MAPE":          mean_absolute_percentage_error(y_test_price, y_pred_price),
                "Median AE (€)": median_absolute_error(y_test_price, y_pred_price),
            }
        }

    comparison_df = pd.DataFrame(
        {name: res["metrics"] for name, res in all_results.items()}
    )

    winning_model_name = min(all_results, key=lambda n: all_results[n]["metrics"]["RMSE (€)"])
    best_pipeline      = all_results[winning_model_name]["pipeline"]
    final_model        = best_pipeline.named_steps["model"]
    feature_names      = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(final_model, "feature_importances_"):
        importances  = final_model.feature_importances_
        metric_name  = "Feature Importance"
    elif hasattr(final_model, "coef_"):
        importances  = np.abs(np.ravel(final_model.coef_))
        metric_name  = "Absolute Coefficient"
    else:
        importances  = np.ones(len(feature_names))
        metric_name  = "N/A"

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    )
    importance_df["Feature"] = (
        importance_df["Feature"].str.replace("num__","",regex=False).str.replace("cat__","",regex=False)
    )
    importance_df["Importance Share"] = importance_df["Importance"] / importance_df["Importance"].sum()

    return {
        "cv_results_df":        cv_results_df,
        "tuning_df":            tuning_df,
        "tuning_comparison_df": tuning_comparison_df,
        "comparison_df":        comparison_df,
        "all_results":          all_results,
        "winning_model_name":   winning_model_name,
        "importance_df":        importance_df,
        "metric_name":          metric_name,
        "grid_objects":         grid_objects,
        "X_test":               X_test,
        "y_test":               y_test,
    }

if os.path.exists(CACHE_FILE):
    print("Loading ML results from cache …")
    ml = joblib.load(CACHE_FILE)
else:
    ml = run_ml_pipeline()
    joblib.dump(ml, CACHE_FILE)
    print("ML results cached.")

cv_results_df        = ml["cv_results_df"]
tuning_df            = ml["tuning_df"]
tuning_comparison_df = ml["tuning_comparison_df"]
comparison_df        = ml["comparison_df"]
all_results          = ml["all_results"]
winning_model_name   = ml["winning_model_name"]
importance_df        = ml["importance_df"]
metric_name          = ml["metric_name"]
grid_objects         = ml["grid_objects"]

# =============================================================================
# HELPER — reusable figure styling
# =============================================================================
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#2d2d2d"),
    margin=dict(t=50, b=40, l=40, r=20),
    colorway=AIRBNB_PALETTE,
)

def apply_theme(fig):
    fig.update_layout(**PLOT_THEME)
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig

def kpi_card(title, value, subtitle="", color=AIRBNB_RED):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="kpi-label"),
            html.H3(value, className="kpi-value", style={"color": color}),
            html.P(subtitle, className="kpi-sub"),
        ]),
        className="kpi-card"
    )

def section_header(title, subtitle=""):
    return html.Div([
        html.H4(title, className="section-title"),
        html.P(subtitle, className="section-sub") if subtitle else None,
        html.Hr(className="section-rule"),
    ], className="section-header")

# =============================================================================
# TAB 1 — DATA LOADING & WRANGLING
# =============================================================================
def build_tab1():
    # Price histogram
    fig_price = make_subplots(rows=1, cols=2,
        subplot_titles=["Nightly Price Distribution","log(Price + 1) Distribution"])
    fig_price.add_trace(go.Histogram(x=df["price"], nbinsx=80, marker_color=AIRBNB_RED,
                                     name="Price"), row=1, col=1)
    fig_price.add_vline(x=df["price"].median(), line_dash="dash", line_color="#484848",
                        annotation_text=f"Median: €{df['price'].median():.0f}", row=1, col=1)
    fig_price.add_trace(go.Histogram(x=df["log_price"], nbinsx=60, marker_color=AIRBNB_TEAL,
                                     name="log Price"), row=1, col=2)
    fig_price.update_layout(title=f"{CITY_NAME} — Price Overview", showlegend=False, **PLOT_THEME)

    # Room type bar + box
    room_counts = df["room_type"].value_counts().reset_index()
    fig_room = make_subplots(rows=1, cols=2,
        subplot_titles=["Listings by Room Type","Price by Room Type"])
    for i, row in room_counts.iterrows():
        fig_room.add_trace(go.Bar(x=[row["room_type"]], y=[row["count"]],
                                  marker_color=AIRBNB_PALETTE[i % 6], showlegend=False), row=1, col=1)
    for i, rt in enumerate(df["room_type"].unique()):
        subset = df[df["room_type"] == rt]["price"]
        fig_room.add_trace(go.Box(y=subset, name=rt, marker_color=AIRBNB_PALETTE[i % 6],
                                  boxpoints=False), row=1, col=2)
    fig_room.update_layout(title=f"{CITY_NAME} — Room Type Overview", **PLOT_THEME)

    # Correlation heatmap
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdYlGn",
                         zmin=-1, zmax=1, title="Feature Correlation Matrix")
    apply_theme(fig_corr)

    return dbc.Container([
        section_header("Data Loading & Wrangling",
                       "Sections 2–3 · Data Acquisition & Cleaning Pipeline"),

        # KPIs
        dbc.Row([
            dbc.Col(kpi_card("Total Listings",    f"{len(df):,}"),            md=2),
            dbc.Col(kpi_card("Total Reviews",     f"{len(reviews_raw):,}",    color=AIRBNB_TEAL), md=2),
            dbc.Col(kpi_card("Neighbourhoods",    str(df["neighbourhood_cleansed"].nunique()), color="#FC642D"), md=2),
            dbc.Col(kpi_card("Median Price",      f"€{df['price'].median():.0f}"), md=2),
            dbc.Col(kpi_card("Features (clean)",  str(len(df.columns)),       color="#767676"), md=2),
            dbc.Col(kpi_card("Price Range",       f"€15 – €1,000",            color="#FFB400"), md=2),
        ], className="mb-4"),

        # Missing values (interactive slider)
        dbc.Row([
            dbc.Col([
                section_header("3.1 Missing-Value Audit"),
                html.Label("Show top N columns:"),
                dcc.Slider(5, 25, 5, value=15, id="missing-n-slider",
                           marks={i: str(i) for i in range(5, 26, 5)}),
                dcc.Graph(id="missing-bar"),
            ])
        ], className="mb-4"),

        # Price distributions
        dbc.Row([dbc.Col([section_header("3.2 Price Distribution"), dcc.Graph(figure=fig_price)])], className="mb-4"),

        # Room types
        dbc.Row([dbc.Col([section_header("3.2 Room Type Overview"), dcc.Graph(figure=fig_room)])], className="mb-4"),

        # Neighbourhood table
        dbc.Row([
            dbc.Col([
                section_header("Neighbourhood Summary"),
                dash_table.DataTable(
                    id="nbhd-table",
                    data=nbhd_summary.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in nbhd_summary.columns],
                    sort_action="native", filter_action="native",
                    page_size=15, style_table={"overflowX":"auto"},
                    style_header={"backgroundColor": AIRBNB_RED, "color":"white","fontWeight":"bold"},
                    style_cell={"fontFamily":"DM Sans, sans-serif","padding":"8px"},
                    style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#fafafa"}],
                )
            ])
        ], className="mb-4"),

        # Correlation matrix
        dbc.Row([dbc.Col([section_header("Feature Correlation Matrix"), dcc.Graph(figure=fig_corr)])]),
    ], fluid=True, className="tab-content")


# =============================================================================
# TAB 2 — EDA
# =============================================================================
def build_tab2():
    neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique())

    return dbc.Container([
        section_header("Exploratory Data Analysis",
                       "Sections 4–5 · SQL Aggregations + Visual Exploration"),

        # ── SQL Aggregations ────────────────────────────────────────────────
        html.H5("SQL Aggregations", className="sub-heading"),

        dbc.Row([
            dbc.Col([
                html.P("Q1 — Which neighbourhoods are most expensive?", className="query-label"),
                dcc.Graph(figure=apply_theme(px.bar(
                    q1.head(20), x="avg_price", y="neighbourhood",
                    orientation="h", color="avg_price",
                    color_continuous_scale=[[0,"#FFB400"],[1,AIRBNB_RED]],
                    title="Q1 — Top 20 Neighbourhoods by Avg Price",
                    labels={"avg_price":"Avg Price (€)","neighbourhood":""},
                ))),
            ], md=6),
            dbc.Col([
                html.P("Q3 — Does the Superhost badge justify a price premium?", className="query-label"),
                dcc.Graph(figure=apply_theme(px.bar(
                    q3, x="room_type", y="avg_price", color="host_status",
                    barmode="group", color_discrete_sequence=[AIRBNB_TEAL, AIRBNB_RED],
                    title="Q3 — Superhost Price Premium by Room Type",
                    labels={"avg_price":"Avg Price (€)","room_type":""},
                ))),
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q2 — Price by room type × neighbourhood (select neighbourhood):", className="query-label"),
                dcc.Dropdown(
                    id="q2-nbhd-dropdown", options=[{"label":n,"value":n} for n in neighbourhoods],
                    value=neighbourhoods[:6], multi=True, placeholder="Select neighbourhoods …"
                ),
                dcc.Graph(id="q2-chart"),
            ]),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q4 — Price step-up per additional guest", className="query-label"),
                dcc.Graph(figure=apply_theme(px.scatter(
                    q4, x="accommodates", y="avg_price", size="n_listings",
                    trendline="ols", color_discrete_sequence=[AIRBNB_RED],
                    title="Q4 — Price vs Guest Capacity",
                    labels={"avg_price":"Avg Price (€)","accommodates":"Guests"},
                ))),
            ], md=6),
            dbc.Col([
                html.P("Q5 — Availability tier vs price", className="query-label"),
                dcc.Graph(figure=apply_theme(px.bar(
                    q5, x="availability_tier", y="avg_price",
                    color="availability_tier", color_discrete_sequence=AIRBNB_PALETTE,
                    title="Q5 — Avg Price by Availability Tier",
                    labels={"avg_price":"Avg Price (€)","availability_tier":""},
                ))),
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q6 — Multi-listing hosts vs single-listing hosts", className="query-label"),
                dcc.Graph(figure=apply_theme(px.bar(
                    q6, x="host_portfolio", y="avg_price",
                    color="host_portfolio", color_discrete_sequence=AIRBNB_PALETTE,
                    title="Q6 — Avg Price by Host Portfolio Size",
                    labels={"avg_price":"Avg Price (€)","host_portfolio":""},
                ))),
            ], md=6),
            dbc.Col([
                html.P("Q7 — Price tier vs review scores", className="query-label"),
                dcc.Graph(figure=apply_theme(px.imshow(
                    q7.set_index("price_tier")[["avg_overall","avg_clean","avg_location","avg_value"]],
                    text_auto=".3f", color_continuous_scale="YlGnBu",
                    title="Q7 — Review Scores by Price Tier",
                ))),
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q8 — Minimum-night policy distribution", className="query-label"),
                dcc.Graph(figure=apply_theme(px.bar(
                    q8, x="min_night_bucket", y="n_listings",
                    color="min_night_bucket", text="pct_of_total",
                    color_discrete_sequence=AIRBNB_PALETTE,
                    title="Q8 — Minimum-Night Policy Buckets",
                    labels={"n_listings":"Listings","min_night_bucket":""},
                ))),
            ]),
        ], className="mb-4"),

        html.Hr(),

        # ── EDA Charts ───────────────────────────────────────────────────────
        html.H5("EDA Visualisations", className="sub-heading"),

        # 5.3 Room type — neighbourhood filtered
        dbc.Row([
            dbc.Col([
                section_header("5.3 Room-Type Breakdown"),
                html.Label("Filter by neighbourhood:"),
                dcc.Dropdown(
                    id="eda-nbhd-dropdown",
                    options=[{"label":"All","value":"ALL"}] + [{"label":n,"value":n} for n in neighbourhoods],
                    value="ALL", clearable=False,
                ),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="eda-room-bar"), md=6),
                    dbc.Col(dcc.Graph(id="eda-room-box"), md=6),
                ]),
            ])
        ], className="mb-4"),

        # 5.4 Monthly reviews
        dbc.Row([
            dbc.Col([
                section_header("5.4 Monthly Review Volume — Demand Proxy"),
                dcc.Graph(id="review-ts-chart"),
                dcc.RangeSlider(
                    id="review-date-slider",
                    min=int(review_ts["date"].dt.year.min()),
                    max=int(review_ts["date"].dt.year.max()),
                    value=[2018, int(review_ts["date"].dt.year.max())],
                    marks={y: str(y) for y in range(2018, int(review_ts["date"].dt.year.max())+1)},
                    step=1,
                ),
            ])
        ], className="mb-4"),

        # 5.5 Superhost
        dbc.Row([
            dbc.Col([
                section_header("5.5 Superhost vs Non-Superhost"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=apply_theme(px.box(
                        superhost_comp, x="superhost_label", y="price",
                        color="superhost_label", color_discrete_sequence=[AIRBNB_TEAL, AIRBNB_RED],
                        title="Price: Superhost vs Non-Superhost",
                        labels={"price":"Nightly Price (€)","superhost_label":""},
                        points=False,
                    ))), md=6),
                    dbc.Col(dcc.Graph(figure=apply_theme(px.box(
                        superhost_comp, x="superhost_label", y="review_scores_rating",
                        color="superhost_label", color_discrete_sequence=[AIRBNB_TEAL, AIRBNB_RED],
                        title="Rating: Superhost vs Non-Superhost",
                        labels={"review_scores_rating":"Review Score","superhost_label":""},
                        points=False,
                    ))), md=6),
                ]),
            ])
        ], className="mb-4"),

        # 5.6 Minimum nights
        dbc.Row([
            dbc.Col([
                section_header("5.6 Minimum Nights Distribution"),
                html.Label("Cap histogram x-axis at (nights):"),
                dcc.Slider(10, 60, 10, value=60, id="minnights-cap-slider",
                           marks={i: str(i) for i in range(10, 61, 10)}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="minnights-hist"), md=6),
                    dbc.Col(dcc.Graph(figure=apply_theme(
                        (lambda mn_band: px.bar(
                            mn_band, x="min_stay_band", y="n_listings",
                            color="min_stay_band", color_discrete_sequence=AIRBNB_PALETTE,
                            title="Minimum-Stay Policy Buckets",
                            labels={"n_listings":"Listings","min_stay_band":""},
                        ))(
                            df[df["minimum_nights"] > 0].assign(
                                min_stay_band=lambda x: pd.cut(
                                    x["minimum_nights"], bins=[0,2,7,29,np.inf],
                                    labels=["1–2 nights","3–7 nights","8–29 nights","30+ nights"]
                                )
                            )["min_stay_band"].value_counts(sort=False)
                            .rename_axis("min_stay_band").reset_index(name="n_listings")
                        )
                    )), md=6),
                ]),
            ])
        ], className="mb-4"),

        # 5.7 Host portfolio
        dbc.Row([
            dbc.Col([
                section_header("5.7 Host Portfolio Size"),
                html.Label("Cap histogram x-axis at (listings):"),
                dcc.Slider(5, 20, 5, value=20, id="portfolio-cap-slider",
                           marks={i: str(i) for i in range(5, 21, 5)}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="portfolio-hist"), md=6),
                    dbc.Col(dcc.Graph(figure=apply_theme(
                        (lambda pb: px.bar(
                            pb, x="host_type", y="n_listings",
                            color="host_type", color_discrete_sequence=AIRBNB_PALETTE,
                            title="Single vs Multi-Listing Hosts",
                            labels={"n_listings":"Listings","host_type":""},
                        ))(
                            df[df["calculated_host_listings_count"] > 0].assign(
                                host_type=lambda x: pd.cut(
                                    x["calculated_host_listings_count"], bins=[0,1,3,10,np.inf],
                                    labels=["1 listing","2–3 listings","4–10 listings","11+ listings"]
                                )
                            )["host_type"].value_counts(sort=False)
                            .rename_axis("host_type").reset_index(name="n_listings")
                        )
                    )), md=6),
                ]),
            ])
        ]),
    ], fluid=True, className="tab-content")


# =============================================================================
# TAB 3 — MACHINE LEARNING
# =============================================================================
def build_tab3():
    winning_metrics = all_results[winning_model_name]["metrics"]
    top20 = importance_df.head(20).copy()

    # CV RMSE chart
    fig_cv_rmse = apply_theme(px.bar(
        cv_results_df.sort_values("CV RMSE Mean"),
        x="CV RMSE Mean", y="Model", orientation="h",
        error_x="CV RMSE Std", color="CV RMSE Mean",
        color_continuous_scale=[[0,AIRBNB_TEAL],[1,AIRBNB_RED]],
        title="Cross-Validated RMSE by Model (lower = better)",
    ))

    # Pre vs post tune
    fig_tune_compare = apply_theme(px.bar(
        tuning_comparison_df.melt(id_vars="Model", value_vars=["Pre-tune CV RMSE","Post-tune CV RMSE"],
                                   var_name="Stage", value_name="RMSE"),
        x="Model", y="RMSE", color="Stage", barmode="group",
        color_discrete_sequence=[AIRBNB_PALETTE[4], AIRBNB_RED],
        title="Pre-Tune vs Post-Tune CV RMSE",
    ))

    # Improvement chart
    fig_improvement = apply_theme(px.bar(
        tuning_comparison_df.sort_values("Improvement", ascending=False),
        x="Improvement", y="Model", orientation="h",
        color="Improvement", color_continuous_scale=[[0,"#f0f0f0"],[1,AIRBNB_TEAL]],
        title="RMSE Improvement from Hyperparameter Tuning",
    ))

    # Feature importance
    fig_feat = apply_theme(px.bar(
        top20, x="Importance", y="Feature", orientation="h",
        color="Importance Share", color_continuous_scale=[[0,"#FFB400"],[1,AIRBNB_RED]],
        title=f"Top 20 Price Drivers — {winning_model_name} ({metric_name})",
    ))
    fig_feat.update_yaxes(autorange="reversed")

    # Final metrics table
    comp_display = comparison_df.reset_index().rename(columns={"index":"Metric"})
    comp_display = comp_display.round(4)

    return dbc.Container([
        section_header("Machine Learning Modelling",
                       "Sections 8–9 · CV · Tuning · Final Evaluation · Feature Importance"),

        # Winning model banner
        dbc.Alert([
            html.Strong(f"🏆 Winning Model: {winning_model_name}  ·  "),
            f"RMSE (€): {winning_metrics['RMSE (€)']:.2f}  ·  "
            f"MAE (€): {winning_metrics['MAE (€)']:.2f}  ·  "
            f"R² (€): {winning_metrics['R² (€)']:.4f}",
        ], color="success", className="mb-4"),

        # CV comparison
        dbc.Row([
            dbc.Col([
                section_header("A — Cross-Validation Comparison"),
                dbc.RadioItems(
                    id="cv-metric-radio",
                    options=[
                        {"label":"RMSE","value":"rmse"},
                        {"label":"MAE", "value":"mae"},
                        {"label":"R²",  "value":"r2"},
                    ],
                    value="rmse", inline=True, className="mb-2",
                ),
                dcc.Graph(id="cv-metric-chart"),
            ])
        ], className="mb-4"),

        # Tuning
        dbc.Row([
            dbc.Col([
                section_header("B — Hyperparameter Tuning"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_tune_compare), md=6),
                    dbc.Col(dcc.Graph(figure=fig_improvement),  md=6),
                ]),
                html.Label("View tuning trajectory for model:"),
                dcc.Dropdown(
                    id="tuning-model-dropdown",
                    options=[{"label":m,"value":m} for m in grid_objects],
                    value=list(grid_objects.keys())[0], clearable=False,
                ),
                dcc.Graph(id="tuning-trajectory-chart"),
            ])
        ], className="mb-4"),

        # Final evaluation table
        dbc.Row([
            dbc.Col([
                section_header("C — Final Hold-Out Test Results"),
                dash_table.DataTable(
                    data=comp_display.to_dict("records"),
                    columns=[{"name":c,"id":c} for c in comp_display.columns],
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor":AIRBNB_RED,"color":"white","fontWeight":"bold"},
                    style_cell={"fontFamily":"DM Sans, sans-serif","padding":"8px","textAlign":"center"},
                    style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#fafafa"}],
                ),
            ])
        ], className="mb-4"),

        # Feature importance
        dbc.Row([
            dbc.Col([
                section_header("D — Feature Importance"),
                html.Label("Show top N features:"),
                dcc.Slider(10, len(importance_df), step=10,
                           value=20, id="feat-n-slider",
                           marks={10:"10",20:"20",30:"30",
                                  min(50,len(importance_df)):str(min(50,len(importance_df)))}),
                dcc.Graph(id="feat-importance-chart"),
            ])
        ]),
    ], fluid=True, className="tab-content")


# =============================================================================
# TAB 4 — INTERACTIVE PRICE EXPLORER
# =============================================================================
def build_tab4():
    neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique())
    room_types     = sorted(df["room_type"].dropna().unique())
    price_max      = int(df["price"].quantile(0.99))

    return dbc.Container([
        section_header("Interactive Price Explorer",
                       "Section 10 · Filter listings live by neighbourhood, room type, price & capacity"),

        dbc.Row([
            # Sidebar filters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Strong("🔍 Filters")),
                    dbc.CardBody([
                        html.Label("Neighbourhood"),
                        dcc.Dropdown(
                            id="exp-nbhd", options=[{"label":n,"value":n} for n in neighbourhoods],
                            value=[], multi=True, placeholder="All neighbourhoods",
                        ),
                        html.Br(),
                        html.Label("Room Type"),
                        dcc.Checklist(
                            id="exp-room",
                            options=[{"label":rt,"value":rt} for rt in room_types],
                            value=room_types, inputStyle={"marginRight":"6px"},
                            labelStyle={"display":"block"},
                        ),
                        html.Br(),
                        html.Label("Price Range (€ / night)"),
                        dcc.RangeSlider(
                            id="exp-price", min=15, max=price_max,
                            value=[15, price_max], step=5,
                            marks={15:"€15", price_max//2:f"€{price_max//2}", price_max:f"€{price_max}"},
                        ),
                        html.Br(),
                        html.Label("Guest Capacity"),
                        dcc.RangeSlider(id="exp-acc", min=1, max=16, value=[1,16], step=1,
                                        marks={i: str(i) for i in range(1, 17, 3)}),
                        html.Br(),
                        html.Label("Host Type"),
                        dcc.RadioItems(
                            id="exp-superhost",
                            options=[
                                {"label":"All","value":"all"},
                                {"label":"Superhost only","value":"1"},
                                {"label":"Non-superhost only","value":"0"},
                            ],
                            value="all", labelStyle={"display":"block"},
                            inputStyle={"marginRight":"6px"},
                        ),
                    ])
                ], className="filter-card"),
            ], md=3),

            # Charts
            dbc.Col([
                dbc.Row(id="exp-kpis", className="mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="exp-price-hist"), md=6),
                    dbc.Col(dcc.Graph(id="exp-scatter"),    md=6),
                ]),
                dbc.Row([dbc.Col(dcc.Graph(id="exp-nbhd-box"))], className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Filtered Listings", className="mt-3"),
                        dash_table.DataTable(
                            id="exp-table",
                            columns=[
                                {"name":"Neighbourhood","id":"neighbourhood_cleansed"},
                                {"name":"Room Type",    "id":"room_type"},
                                {"name":"Price (€)",    "id":"price"},
                                {"name":"Accommodates", "id":"accommodates"},
                                {"name":"Bedrooms",     "id":"bedrooms"},
                                {"name":"Rating",       "id":"review_scores_rating"},
                                {"name":"Superhost",    "id":"is_superhost"},
                            ],
                            page_size=20, sort_action="native",
                            style_table={"overflowX":"auto"},
                            style_header={"backgroundColor":AIRBNB_RED,"color":"white","fontWeight":"bold"},
                            style_cell={"fontFamily":"DM Sans, sans-serif","padding":"6px"},
                            style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#fafafa"}],
                        )
                    ])
                ], className="mt-3"),
            ], md=9),
        ]),
    ], fluid=True, className="tab-content")


# =============================================================================
# TAB 5 — STORYTELLING & COMMUNICATION
# =============================================================================
def build_tab5():
    win_m = all_results[winning_model_name]["metrics"]
    sh_med = superhost_comp.groupby("superhost_label")["price"].median()
    top_feature = importance_df.iloc[0]["Feature"]

    return dbc.Container([
        section_header("Storytelling & Communication",
                       "Section 11 · Key findings narrated for a non-technical audience"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("🏡 What Makes a Dublin Airbnb Expensive?", className="story-h"),
                    dcc.Markdown(f"""
Dublin's Airbnb market is primarily driven by **listing size and capacity** rather than by
host reputation or review activity. The single strongest predictor of nightly price is
**{top_feature}**, confirming that guests are essentially paying for space.

Entire home listings command the highest prices and the widest price spread — a one-bedroom
city-centre apartment and a five-bedroom suburban house are both classified as "Entire home"
but differ by hundreds of euros per night. By contrast, private and shared rooms cluster
tightly at the budget end of the market.

Neighbourhood matters, but less than you might expect once room type and capacity are
accounted for. Central neighbourhoods carry a modest premium, while outer areas show
competitive pricing even for comparable properties.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("📊 Key Numbers at a Glance", className="story-h"),
                    dbc.Row([
                        dbc.Col(kpi_card("Median Price", f"€{df['price'].median():.0f}/night"), md=3),
                        dbc.Col(kpi_card("Superhost Median", f"€{sh_med.get('Superhost',0):.0f}",
                                         color=AIRBNB_TEAL), md=3),
                        dbc.Col(kpi_card("Non-SH Median", f"€{sh_med.get('Non-superhost',0):.0f}",
                                         color="#FC642D"), md=3),
                        dbc.Col(kpi_card("Best Model MAE", f"€{win_m['MAE (€)']:.0f}",
                                         color="#FFB400"), md=3),
                    ]),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("🏅 The Superhost Paradox", className="story-h"),
                    dcc.Markdown(f"""
Superhosts are **more affordable, not more expensive**. The median nightly price for a
Superhost listing is €{sh_med.get('Superhost', 0):.0f}, compared to €{sh_med.get('Non-superhost', 0):.0f}
for non-Superhosts. Guests staying with Superhosts trade a small price premium for
meaningfully higher review scores, particularly on cleanliness and communication.

This suggests the Superhost badge reflects hospitality quality rather than luxury pricing —
a finding that has direct implications for how guests should weight the badge in their
booking decisions.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("📈 Demand Signal: Monthly Reviews", className="story-h"),
                    dcc.Markdown("""
Review volume acts as a **demand proxy** (approximately 50–70% of stays generate a review).
The time-series chart in the EDA tab clearly shows:

- A **sharp collapse in March 2020** marking the onset of COVID-19 travel restrictions.
- A **strong recovery through 2022–2023**, with review volumes returning to and exceeding
  pre-pandemic levels by late 2023.
- Clear **seasonal cycles** with summer peaks, consistent with Dublin's tourism calendar.

The post-pandemic growth in listings suggests both renewed tourist demand and an increasing
commercialisation of the platform by multi-listing professional hosts.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5(f"🤖 Model Performance — {winning_model_name}", className="story-h"),
                    dcc.Markdown(f"""
After testing **9 regression models** with 5-fold cross-validation and hyperparameter tuning,
**{winning_model_name}** delivered the best hold-out test performance:

| Metric | Value |
|--------|-------|
| RMSE (€) | €{win_m['RMSE (€)']:.2f} |
| MAE (€)  | €{win_m['MAE (€)']:.2f} |
| R² (€)   | {win_m['R² (€)']:.4f} |
| MAPE     | {win_m['MAPE']:.1%} |

In practical terms, the model's median prediction error is roughly **€{win_m['Median AE (€)']:.0f}** —
meaning that for a typical Dublin listing the predicted price is within €{win_m['Median AE (€)']:.0f}
of the actual listed price. This level of accuracy is useful for pricing guidance but should
not replace host judgement on unique property characteristics.
                    """, className="story-body"),
                ]), className="story-card"),
            ])
        ]),
    ], fluid=True, className="tab-content")


# =============================================================================
# TAB 6 — CRITICAL THINKING & REFLECTION
# =============================================================================
def build_tab6():
    return dbc.Container([
        section_header("Critical Thinking & Reflection",
                       "Section 12 · Limitations, Ethics, and Future Work"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("⚠️ Data Limitations", className="story-h"),
                    dcc.Markdown("""
**Inside Airbnb is a snapshot, not ground truth.**  The dataset captures listed prices
and features at a single point in time, not actual booking prices, occupancy rates, or
revenue. Hosts routinely adjust prices dynamically — the listed price on a given day may
not reflect what a guest actually paid.

**Review scores are left-censored.** Very few listings have scores below 4.0 because
unhappy guests often leave no review. This compresses the rating scale and limits the
discriminating power of review features.

**No calendar data in the model.** The calendar file (availability_365) was used in
exploratory analysis but not as a time-varying feature. Incorporating actual blocked/
booked dates could substantially improve predictive accuracy.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("🔬 Modelling Limitations", className="story-h"),
                    dcc.Markdown("""
**Target leakage risk.** Review scores (cleanliness, location, value) are collected
*after* a guest stays and are correlated with price. Including them as features may
introduce subtle target leakage if higher-priced listings systematically attract
higher-scoring reviews independent of actual quality.

**No spatial autocorrelation.** The model treats neighbourhood as a categorical variable.
A spatial model (e.g., geographically weighted regression or a spatial lag term) could
capture micro-location effects that a one-hot-encoded neighbourhood label cannot.

**Log-price target.** Predicting log(price+1) improves residual normality but means that
errors are multiplicative in the original price space. A €10 error at €50/night is much
more significant than a €10 error at €500/night; MAPE partially corrects for this but
is sensitive to very low prices.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("⚖️ Ethical Considerations", className="story-h"),
                    dcc.Markdown("""
**Housing displacement.** The prevalence of multi-listing professional hosts (≈54% of
listings belong to hosts with more than one property) raises questions about the
platform's role in reducing long-term rental supply in Dublin — a city with a severe
housing shortage. A price-prediction model that helps professional hosts optimise revenue
may inadvertently accelerate this effect.

**Neighbourhood-level pricing inequality.** The model uses neighbourhood as a feature,
which means it learns and perpetuates existing geographic price disparities. Care should
be taken not to present the model's outputs as normative pricing benchmarks.

**Transparency.** Guests do not know that algorithmic pricing tools are being used.
Disclosure norms for AI-assisted pricing in the short-term rental market remain
undeveloped.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("🚀 Future Work", className="story-h"),
                    dcc.Markdown("""
1. **Choropleth map** — overlay median prices and listing density on the Dublin
   neighbourhood GeoJSON already loaded in the pipeline.
2. **Time-series forecasting** — model monthly demand (review volume) using Prophet or
   LSTM to predict seasonal peaks for capacity planning.
3. **SHAP interaction plots** — decompose individual predictions to explain *why* a
   specific listing is priced above or below the model's expectation.
4. **Live calendar integration** — scrape current calendar data to add real-time
   availability and dynamic pricing signals as features.
5. **Natural language features** — apply TF-IDF or sentence embeddings to listing
   descriptions and host profiles to capture quality signals not reflected in structured
   fields.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("💭 Process Reflection", className="story-h"),
                    dcc.Markdown("""
The most time-consuming phase was not modelling but **data cleaning**. The raw Inside
Airbnb listings file contains price strings with currency symbols, boolean columns
encoded as "t"/"f", percentage columns stored as strings, and free-text fields like
`bathrooms_text` that require regex extraction. Investing in a robust, function-based
cleaning pipeline (`.pipe()` chains) paid dividends when the same logic was reused in
the ML preprocessing step.

The decision to use `log(price+1)` as the modelling target was validated by inspecting
residuals — the raw price distribution is heavily right-skewed, and models trained on it
performed poorly at the budget end of the market. The log transform brought the target
distribution close to normal and improved all model metrics substantially.

Finally, caching the GridSearchCV results with `joblib` was essential for making the
dashboard practical to deploy — re-running the full tuning grid on every app start would
take 20–30 minutes.
                    """, className="story-body"),
                ]), className="story-card"),
            ])
        ]),
    ], fluid=True, className="tab-content")


# =============================================================================
# APP LAYOUT
# =============================================================================
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap",
    ],
    suppress_callback_exceptions=True,
)
server = app.server  # expose for Render / gunicorn

app.layout = html.Div([
    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("airbnb", className="logo-text"),
            html.Span(" analytics", className="logo-sub"),
        ], className="logo"),
        html.Div([
            html.H1(f"Dublin Airbnb Dashboard", className="header-title"),
            html.P("Data: Inside Airbnb · Dublin · Group 7", className="header-sub"),
        ], className="header-text"),
    ], className="dashboard-header"),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id="main-tabs", value="tab1", className="main-tabs", children=[
        dcc.Tab(label="① Data Wrangling",       value="tab1", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="② EDA",                  value="tab2", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="③ Machine Learning",     value="tab3", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="④ Price Explorer",       value="tab4", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="⑤ Storytelling",         value="tab5", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="⑥ Reflection",           value="tab6", className="tab", selected_className="tab--selected"),
    ]),
    html.Div(id="tab-content"),
], className="app-shell")

# =============================================================================
# CALLBACKS
# =============================================================================

# Render active tab
@app.callback(Output("tab-content","children"), Input("main-tabs","value"))
def render_tab(tab):
    if tab == "tab1": return build_tab1()
    if tab == "tab2": return build_tab2()
    if tab == "tab3": return build_tab3()
    if tab == "tab4": return build_tab4()
    if tab == "tab5": return build_tab5()
    if tab == "tab6": return build_tab6()

# Tab 1 — missing value slider
@app.callback(Output("missing-bar","figure"), Input("missing-n-slider","value"))
def update_missing_bar(n):
    data = missing_report.head(n)
    fig = px.bar(data, x="pct", y="column", orientation="h",
                 color="pct", color_continuous_scale=[[0,"#FFB400"],[1,AIRBNB_RED]],
                 title=f"Top {n} Columns by % Missing",
                 labels={"pct":"% Missing","column":""})
    return apply_theme(fig)

# Tab 2 — Q2 neighbourhood dropdown
@app.callback(Output("q2-chart","figure"), Input("q2-nbhd-dropdown","value"))
def update_q2(selected):
    if not selected:
        data = q2
    else:
        data = q2[q2["neighbourhood"].isin(selected)]
    fig = px.bar(data, x="neighbourhood", y="avg_price", color="room_type",
                 barmode="group", color_discrete_sequence=AIRBNB_PALETTE,
                 title="Q2 — Avg Price by Room Type × Neighbourhood",
                 labels={"avg_price":"Avg Price (€)","neighbourhood":""})
    return apply_theme(fig)

# Tab 2 — EDA room type charts
@app.callback(
    Output("eda-room-bar","figure"),
    Output("eda-room-box","figure"),
    Input("eda-nbhd-dropdown","value"),
)
def update_eda_room(nbhd):
    subset = df if nbhd == "ALL" else df[df["neighbourhood_cleansed"] == nbhd]
    rc = subset["room_type"].value_counts().reset_index()
    fig_bar = apply_theme(px.bar(rc, x="room_type", y="count", color="room_type",
                                  color_discrete_sequence=AIRBNB_PALETTE,
                                  title="Listings by Room Type",
                                  labels={"count":"Listings","room_type":""}))
    fig_box = apply_theme(px.box(subset, x="room_type", y="price", color="room_type",
                                  color_discrete_sequence=AIRBNB_PALETTE,
                                  title="Price by Room Type",
                                  labels={"price":"Nightly Price (€)","room_type":""},
                                  points=False))
    return fig_bar, fig_box

# Tab 2 — review time-series date range
@app.callback(Output("review-ts-chart","figure"), Input("review-date-slider","value"))
def update_review_ts(years):
    data = review_ts[(review_ts["date"].dt.year >= years[0]) &
                     (review_ts["date"].dt.year <= years[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["review_count"],
                              fill="tozeroy", line=dict(color=AIRBNB_RED,width=2),
                              name="Reviews / Month"))
    fig.add_vrect(x0="2020-03-01", x1="2021-06-01", fillcolor="gray",
                  opacity=0.12, annotation_text="COVID-19 restrictions", line_width=0)
    fig.update_layout(title="Monthly Review Volume (demand proxy)", **PLOT_THEME)
    return fig

# Tab 2 — min nights histogram cap
@app.callback(Output("minnights-hist","figure"), Input("minnights-cap-slider","value"))
def update_minnights(cap):
    data = df[(df["minimum_nights"] > 0) & (df["minimum_nights"] <= cap)]
    fig = apply_theme(px.histogram(data, x="minimum_nights", nbins=cap,
                                    color_discrete_sequence=[AIRBNB_RED],
                                    title=f"Min Nights Distribution (cap={cap})",
                                    labels={"minimum_nights":"Minimum Nights"}))
    fig.add_vline(x=30, line_dash="dash", line_color="#767676",
                  annotation_text="30-night threshold")
    return fig

# Tab 2 — host portfolio cap
@app.callback(Output("portfolio-hist","figure"), Input("portfolio-cap-slider","value"))
def update_portfolio(cap):
    data = df[(df["calculated_host_listings_count"] > 0) &
              (df["calculated_host_listings_count"] <= cap)]
    return apply_theme(px.histogram(data, x="calculated_host_listings_count", nbins=cap,
                                     color_discrete_sequence=[AIRBNB_TEAL],
                                     title=f"Listings per Host (cap={cap})",
                                     labels={"calculated_host_listings_count":"Listings per Host"}))

# Tab 3 — CV metric toggle
@app.callback(Output("cv-metric-chart","figure"), Input("cv-metric-radio","value"))
def update_cv_chart(metric):
    col_map = {"rmse":("CV RMSE Mean","CV RMSE Std"),
               "mae": ("CV MAE Mean", "CV MAE Std"),
               "r2":  ("CV R2 Mean",  "CV R2 Std")}
    mean_col, std_col = col_map[metric]
    asc = metric != "r2"
    data = cv_results_df.sort_values(mean_col, ascending=asc)
    title_map = {"rmse":"CV RMSE (lower = better)","mae":"CV MAE (lower = better)","r2":"CV R² (higher = better)"}
    fig = apply_theme(px.bar(
        data, x=mean_col, y="Model", orientation="h", error_x=std_col,
        color=mean_col, color_continuous_scale=[[0,AIRBNB_TEAL],[1,AIRBNB_RED]],
        title=f"Model Comparison — {title_map[metric]}",
    ))
    return fig

# Tab 3 — tuning trajectory dropdown
@app.callback(Output("tuning-trajectory-chart","figure"), Input("tuning-model-dropdown","value"))
def update_tuning_traj(model_name):
    grid = grid_objects.get(model_name)
    if grid is None:
        return go.Figure()
    cv_res = pd.DataFrame(grid.cv_results_)
    cv_res["RMSE"] = -cv_res["mean_test_score"]
    top15 = cv_res.nsmallest(15,"RMSE").copy().reset_index(drop=True)
    top15["Rank"] = [f"Rank {i+1}" for i in range(len(top15))]
    fig = apply_theme(px.bar(
        top15, x="RMSE", y="Rank", orientation="h",
        color="RMSE", color_continuous_scale=[[0,AIRBNB_TEAL],[1,AIRBNB_RED]],
        title=f"Top 15 Hyperparameter Configs — {model_name}",
    ))
    fig.update_yaxes(autorange="reversed")
    return fig

# Tab 3 — feature importance slider
@app.callback(Output("feat-importance-chart","figure"), Input("feat-n-slider","value"))
def update_feat(n):
    data = importance_df.head(n)
    fig = apply_theme(px.bar(
        data, x="Importance", y="Feature", orientation="h",
        color="Importance Share", color_continuous_scale=[[0,"#FFB400"],[1,AIRBNB_RED]],
        title=f"Top {n} Price Drivers — {winning_model_name} ({metric_name})",
    ))
    fig.update_yaxes(autorange="reversed")
    return fig

# Tab 4 — Price Explorer
@app.callback(
    Output("exp-kpis",        "children"),
    Output("exp-price-hist",  "figure"),
    Output("exp-scatter",     "figure"),
    Output("exp-nbhd-box",    "figure"),
    Output("exp-table",       "data"),
    Input("exp-nbhd",         "value"),
    Input("exp-room",         "value"),
    Input("exp-price",        "value"),
    Input("exp-acc",          "value"),
    Input("exp-superhost",    "value"),
)
def update_explorer(nbhds, room_types_sel, price_range, acc_range, superhost_val):
    sub = df.copy()
    if nbhds:
        sub = sub[sub["neighbourhood_cleansed"].isin(nbhds)]
    if room_types_sel:
        sub = sub[sub["room_type"].isin(room_types_sel)]
    sub = sub[(sub["price"] >= price_range[0]) & (sub["price"] <= price_range[1])]
    sub = sub[(sub["accommodates"] >= acc_range[0]) & (sub["accommodates"] <= acc_range[1])]
    if superhost_val == "1":
        sub = sub[sub["is_superhost"] == 1]
    elif superhost_val == "0":
        sub = sub[sub["is_superhost"] == 0]

    kpis = dbc.Row([
        dbc.Col(kpi_card("Listings",     f"{len(sub):,}"),                         md=3),
        dbc.Col(kpi_card("Median Price", f"€{sub['price'].median():.0f}" if len(sub) else "—",
                          color=AIRBNB_TEAL), md=3),
        dbc.Col(kpi_card("Avg Rating",   f"{sub['review_scores_rating'].mean():.2f}" if len(sub) else "—",
                          color="#FC642D"), md=3),
        dbc.Col(kpi_card("Avg Capacity", f"{sub['accommodates'].mean():.1f} guests" if len(sub) else "—",
                          color="#FFB400"), md=3),
    ])

    if len(sub) == 0:
        empty = go.Figure()
        return kpis, empty, empty, empty, []

    fig_hist = apply_theme(px.histogram(sub, x="price", nbins=60,
                                         color_discrete_sequence=[AIRBNB_RED],
                                         title="Price Distribution (filtered)",
                                         labels={"price":"Nightly Price (€)"}))

    fig_scat = apply_theme(px.scatter(sub.sample(min(1000, len(sub)), random_state=1),
                                       x="accommodates", y="price", color="room_type",
                                       color_discrete_sequence=AIRBNB_PALETTE, opacity=0.6,
                                       title="Price vs Capacity",
                                       labels={"price":"Nightly Price (€)","accommodates":"Guests"}))

    top_n = (sub["neighbourhood_cleansed"].value_counts().head(15).index.tolist())
    sub_top = sub[sub["neighbourhood_cleansed"].isin(top_n)]
    fig_box = apply_theme(px.box(sub_top, x="neighbourhood_cleansed", y="price",
                                  color="neighbourhood_cleansed",
                                  color_discrete_sequence=AIRBNB_PALETTE,
                                  title="Price by Neighbourhood (top 15)",
                                  labels={"price":"Nightly Price (€)","neighbourhood_cleansed":""},
                                  points=False))
    fig_box.update_layout(showlegend=False)

    tbl_cols = ["neighbourhood_cleansed","room_type","price",
                "accommodates","bedrooms","review_scores_rating","is_superhost"]
    tbl_data = sub[[c for c in tbl_cols if c in sub.columns]].head(200).round(2).to_dict("records")

    return kpis, fig_hist, fig_scat, fig_box, tbl_data


# =============================================================================
# INLINE CSS (injected via assets — kept here for single-file portability)
# =============================================================================
app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Dublin Airbnb Dashboard</title>
{%favicon%}
{%css%}
<style>
  :root {
    --red:    #FF5A5F;
    --teal:   #00A699;
    --orange: #FC642D;
    --dark:   #484848;
    --grey:   #767676;
    --gold:   #FFB400;
    --bg:     #f7f7f7;
    --white:  #ffffff;
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); font-family: 'DM Sans', sans-serif; margin: 0; }

  .app-shell { min-height: 100vh; }

  /* Header */
  .dashboard-header {
    background: linear-gradient(135deg, var(--dark) 0%, #2a2a2a 100%);
    color: white; padding: 20px 32px; display: flex; align-items: center; gap: 24px;
    border-bottom: 4px solid var(--red);
  }
  .logo { line-height: 1; }
  .logo-text { font-family: 'DM Serif Display', serif; font-size: 28px; color: var(--red); }
  .logo-sub  { font-size: 14px; color: var(--grey); letter-spacing: 2px; text-transform: uppercase; }
  .header-title { margin: 0; font-size: 22px; font-weight: 600; color: white; }
  .header-sub   { margin: 0; font-size: 12px; color: #aaa; }

  /* Tabs */
  .main-tabs { background: var(--dark); padding: 0 24px; border: none !important; }
  .tab { background: transparent !important; border: none !important;
         color: #bbb !important; padding: 14px 18px !important;
         font-size: 13px; font-weight: 500; letter-spacing: 0.3px; }
  .tab:hover { color: white !important; }
  .tab--selected { color: white !important; border-bottom: 3px solid var(--red) !important;
                   background: rgba(255,90,95,0.12) !important; }

  /* Tab content */
  .tab-content { padding: 28px 24px; }

  /* Section headers */
  .section-header { margin-bottom: 16px; }
  .section-title  { font-family: 'DM Serif Display', serif; font-size: 20px;
                    color: var(--dark); margin-bottom: 2px; }
  .section-sub    { font-size: 12px; color: var(--grey); margin-bottom: 6px; }
  .section-rule   { border: none; border-top: 2px solid var(--red);
                    opacity: 0.3; margin: 6px 0 12px; }
  .sub-heading    { font-size: 16px; font-weight: 600; color: var(--dark);
                    margin: 20px 0 10px; padding-left: 10px;
                    border-left: 4px solid var(--red); }

  /* KPI cards */
  .kpi-card   { border: none; border-radius: var(--radius); background: var(--white);
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 100%; }
  .kpi-label  { font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                color: var(--grey); margin-bottom: 4px; }
  .kpi-value  { font-size: 24px; font-weight: 700; margin: 0; }
  .kpi-sub    { font-size: 11px; color: var(--grey); margin-top: 2px; }

  /* Query labels */
  .query-label { font-size: 12px; font-style: italic; color: var(--grey); margin-bottom: 4px; }

  /* Storytelling */
  .story-card { border: none; border-radius: var(--radius); background: var(--white);
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .story-h    { font-family: 'DM Serif Display', serif; font-size: 18px; color: var(--dark); }
  .story-body { font-size: 14px; line-height: 1.7; color: #444; }

  /* Filter card */
  .filter-card { border: none; border-radius: var(--radius); background: var(--white);
                 box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 100%; }
  .filter-card .card-header { background: var(--dark); color: white;
                               border-radius: var(--radius) var(--radius) 0 0; }

  /* Plotly chart background */
  .js-plotly-plot .plotly { border-radius: var(--radius); }
  .dash-graph { background: var(--white); border-radius: var(--radius);
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); padding: 4px; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
