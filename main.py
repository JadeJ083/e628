import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandasql import sqldf

from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Brand palette ─────────────────────────────────────────────────────────────
AIRBNB_PALETTE = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676", "#FFB400"]
AIRBNB_RED = "#FF5A5F"
AIRBNB_TEAL = "#00A699"
CITY_NAME = "Dublin"

# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================
DATA_DIR = "data"
LISTINGS_PATH = os.path.join(DATA_DIR, "listings.csv.gz")
REVIEWS_PATH = os.path.join(DATA_DIR, "reviews.csv")
CHOROPLETH_MAP_PATH = os.path.join(DATA_DIR, "dublin_choropleth_price.html")
CLICKABLE_MAP_PATH = os.path.join(DATA_DIR, "dublin_listings_clickable.html")

print("Loading local data ...")
try:
    listings_raw = pd.read_csv(LISTINGS_PATH, low_memory=False, compression="gzip")
    reviews_raw = pd.read_csv(REVIEWS_PATH, low_memory=False)
    print(f"Listings: {len(listings_raw):,} rows × {listings_raw.shape[1]} cols")
    print(f"Reviews: {len(reviews_raw):,} rows")
except Exception as e:
    raise RuntimeError(f"Local data load failed. Check files in /data.\n{e}")

# =============================================================================
# SECTION 2 — DATA CLEANING
# =============================================================================
def parse_price(df):
    if "price" not in df.columns:
        return df
    return df.assign(
        price=lambda x: (
            x["price"].astype(str)
            .str.replace(r"[\$\u20ac,]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
    )

def parse_numeric_cols(df):
    cols = [
        "review_scores_rating", "review_scores_cleanliness",
        "review_scores_communication", "review_scores_location",
        "review_scores_value", "accommodates", "bedrooms", "beds",
        "minimum_nights", "number_of_reviews", "availability_365",
        "calculated_host_listings_count", "latitude", "longitude"
    ]
    existing = [c for c in cols if c in df.columns]
    return df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in existing})

def add_derived_features(df):
    return df.assign(
        price_per_person=lambda x: x["price"] / x["accommodates"].replace(0, np.nan)
        if "accommodates" in x.columns else np.nan,
        host_age_years=lambda x: (
            (pd.Timestamp.now() - pd.to_datetime(x.get("host_since"), errors="coerce"))
            .dt.days / 365
        ),
        host_identity_verified=lambda x: (
            x.get("host_identity_verified", pd.Series("f", index=x.index)).map({"t": 1, "f": 0})
        ),
        is_superhost=lambda x: (
            x.get("host_is_superhost", pd.Series("f", index=x.index)).map({"t": 1, "f": 0})
        ),
        log_price=lambda x: np.log1p(x["price"])
    )

def handle_missing_values(df):
    needed = [c for c in ["latitude", "longitude", "price"] if c in df.columns]
    return df.dropna(subset=needed)

def filter_price_outliers(df, lo=15, hi=1000):
    if "price" not in df.columns:
        return df
    return df.query("@lo <= price <= @hi")

print("Cleaning data ...")
df = (
    listings_raw
    .pipe(parse_price)
    .pipe(parse_numeric_cols)
    .pipe(add_derived_features)
    .pipe(handle_missing_values)
    .pipe(filter_price_outliers)
    .reset_index(drop=True)
)
print(f"Clean dataset: {len(df):,} listings")

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
if "neighbourhood_cleansed" in df.columns:
    nbhd_summary = (
        df.groupby("neighbourhood_cleansed", as_index=False)
        .agg(
            listings=("price", "count"),
            median_price=("price", "median"),
            avg_rating=("review_scores_rating", "mean"),
            pct_superhost=("is_superhost", "mean"),
            avg_host_tenure=("host_age_years", "mean"),
        )
        .sort_values("median_price", ascending=False)
        .round(2)
    )
else:
    nbhd_summary = pd.DataFrame(columns=["neighbourhood_cleansed", "listings", "median_price"])

# Correlation matrix
corr_cols = [
    "price", "accommodates", "bedrooms", "beds", "minimum_nights",
    "number_of_reviews", "review_scores_rating", "review_scores_cleanliness",
    "review_scores_location", "review_scores_value",
    "host_age_years", "is_superhost", "calculated_host_listings_count"
]
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].dropna().corr() if corr_cols else pd.DataFrame()

# =============================================================================
# SECTION 3 — SQL AGGREGATIONS
# =============================================================================
print("Running SQL aggregations ...")
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
""") if "neighbourhood_cleansed" in df.columns else pd.DataFrame()

q2 = sql("""
    SELECT neighbourhood_cleansed AS neighbourhood, room_type,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(review_scores_rating),2) AS avg_rating
    FROM df GROUP BY neighbourhood_cleansed, room_type
    HAVING n_listings >= 5 ORDER BY neighbourhood, avg_price DESC
""") if {"neighbourhood_cleansed", "room_type"}.issubset(df.columns) else pd.DataFrame()

q3 = sql("""
    SELECT room_type,
           CASE WHEN is_superhost=1 THEN 'Superhost' ELSE 'Non-superhost' END AS host_status,
           COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(review_scores_rating),2) AS avg_rating,
           ROUND(AVG(number_of_reviews),1) AS avg_reviews
    FROM df GROUP BY room_type, host_status ORDER BY room_type, host_status
""") if "room_type" in df.columns else pd.DataFrame()

q4 = sql("""
    SELECT accommodates, COUNT(*) AS n_listings,
           ROUND(AVG(price),2) AS avg_price,
           ROUND(AVG(bedrooms),1) AS avg_bedrooms,
           ROUND(AVG(beds),1) AS avg_beds
    FROM df WHERE accommodates <= 10
    GROUP BY accommodates ORDER BY accommodates
""") if "accommodates" in df.columns else pd.DataFrame()

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
""") if "availability_365" in df.columns else pd.DataFrame()

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
""") if "calculated_host_listings_count" in df.columns else pd.DataFrame()

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
""") if "review_scores_rating" in df.columns else pd.DataFrame()

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
""") if "minimum_nights" in df.columns else pd.DataFrame()

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
) if "date" in reviews_raw.columns else pd.DataFrame(columns=["date", "review_count"])

# Superhost comparison
superhost_comp = (
    df.assign(superhost_label=lambda x: x["is_superhost"].map({1: "Superhost", 0: "Non-superhost"}))
    .dropna(subset=["is_superhost", "price", "review_scores_rating"])
) if {"is_superhost", "price", "review_scores_rating"}.issubset(df.columns) else pd.DataFrame()

# =============================================================================
# MAP HTML LOADING
# =============================================================================
def load_html_map(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

choropleth_map_html = load_html_map(CHOROPLETH_MAP_PATH)
clickable_map_html = load_html_map(CLICKABLE_MAP_PATH)

# =============================================================================
# HELPER — figure styling
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

def map_card(title, html_content, fallback_text):
    if html_content is None:
        body = dbc.Alert(fallback_text, color="warning", className="mb-0")
    else:
        body = html.Iframe(
            srcDoc=html_content,
            style={"width": "100%", "height": "620px", "border": "none", "borderRadius": "10px"},
        )

    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="story-h mb-3"),
            body
        ]),
        className="story-card"
    )

# =============================================================================
# TAB 1 — DATA WRANGLING + EDA + MAPS
# =============================================================================
def build_tab1():
    fig_price = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Nightly Price Distribution", "log(Price + 1) Distribution"]
    )
    fig_price.add_trace(
        go.Histogram(x=df["price"], nbinsx=80, marker_color=AIRBNB_RED, name="Price"),
        row=1, col=1
    )
    fig_price.add_vline(
        x=df["price"].median(),
        line_dash="dash",
        line_color="#484848",
        annotation_text=f"Median: €{df['price'].median():.0f}",
        row=1, col=1
    )
    fig_price.add_trace(
        go.Histogram(x=df["log_price"], nbinsx=60, marker_color=AIRBNB_TEAL, name="log Price"),
        row=1, col=2
    )
    fig_price.update_layout(title=f"{CITY_NAME} — Price Overview", showlegend=False, **PLOT_THEME)

    room_counts = (
        df["room_type"]
        .value_counts()
        .rename_axis("room_type")
        .reset_index(name="count")
    ) if "room_type" in df.columns else pd.DataFrame(columns=["room_type", "count"])

    fig_room = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Listings by Room Type", "Price by Room Type"]
    )

    for i, row in room_counts.iterrows():
        fig_room.add_trace(
            go.Bar(
                x=[row["room_type"]],
                y=[row["count"]],
                marker_color=AIRBNB_PALETTE[i % len(AIRBNB_PALETTE)],
                showlegend=False
            ),
            row=1, col=1
        )

    if "room_type" in df.columns:
        for i, rt in enumerate(df["room_type"].dropna().unique()):
            subset = df[df["room_type"] == rt]["price"]
            fig_room.add_trace(
                go.Box(
                    y=subset,
                    name=rt,
                    marker_color=AIRBNB_PALETTE[i % len(AIRBNB_PALETTE)],
                    boxpoints=False
                ),
                row=1, col=2
            )

    fig_room.update_layout(title=f"{CITY_NAME} — Room Type Overview", **PLOT_THEME)

    fig_corr = go.Figure()
    if not corr_matrix.empty:
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Matrix"
        )
        apply_theme(fig_corr)

    neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique()) if "neighbourhood_cleansed" in df.columns else []

    return dbc.Container([
        section_header("Data Wrangling & EDA", "Data acquisition, cleaning, SQL summaries, visual exploration, and spatial analysis"),

        dbc.Row([
            dbc.Col(kpi_card("Total Listings", f"{len(df):,}"), md=2),
            dbc.Col(kpi_card("Total Reviews", f"{len(reviews_raw):,}", color=AIRBNB_TEAL), md=2),
            dbc.Col(kpi_card("Neighbourhoods", str(df["neighbourhood_cleansed"].nunique()) if "neighbourhood_cleansed" in df.columns else "—", color="#FC642D"), md=2),
            dbc.Col(kpi_card("Median Price", f"€{df['price'].median():.0f}"), md=2),
            dbc.Col(kpi_card("Columns", str(len(df.columns)), color="#767676"), md=2),
            dbc.Col(kpi_card("Price Range", "€15 – €1,000", color="#FFB400"), md=2),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_header("Missing-Value Audit"),
                html.Label("Show top N columns:"),
                dcc.Slider(5, 25, 5, value=15, id="missing-n-slider",
                           marks={i: str(i) for i in range(5, 26, 5)}),
                dcc.Graph(id="missing-bar"),
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([section_header("Price Distribution"), dcc.Graph(figure=fig_price)])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([section_header("Room Type Overview"), dcc.Graph(figure=fig_room)])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_header("Neighbourhood Summary"),
                dash_table.DataTable(
                    data=nbhd_summary.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in nbhd_summary.columns],
                    sort_action="native",
                    filter_action="native",
                    page_size=15,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": AIRBNB_RED, "color": "white", "fontWeight": "bold"},
                    style_cell={"fontFamily": "DM Sans, sans-serif", "padding": "8px"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
                )
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([section_header("Feature Correlation Matrix"), dcc.Graph(figure=fig_corr)])
        ], className="mb-4"),

        html.Hr(),
        html.H5("SQL Aggregations", className="sub-heading"),

        dbc.Row([
            dbc.Col([
                html.P("Q1 — Which neighbourhoods are most expensive?", className="query-label"),
                dcc.Graph(
                    figure=apply_theme(px.bar(
                        q1.head(20), x="avg_price", y="neighbourhood",
                        orientation="h", color="avg_price",
                        color_continuous_scale=[[0, "#FFB400"], [1, AIRBNB_RED]],
                        title="Q1 — Top 20 Neighbourhoods by Avg Price",
                        labels={"avg_price": "Avg Price (€)", "neighbourhood": ""}
                    )) if not q1.empty else go.Figure()
                ),
            ], md=6),
            dbc.Col([
                html.P("Q3 — Does the Superhost badge justify a price premium?", className="query-label"),
                dcc.Graph(
                    figure=apply_theme(px.bar(
                        q3, x="room_type", y="avg_price", color="host_status",
                        barmode="group", color_discrete_sequence=[AIRBNB_TEAL, AIRBNB_RED],
                        title="Q3 — Superhost Price Premium by Room Type",
                        labels={"avg_price": "Avg Price (€)", "room_type": ""}
                    )) if not q3.empty else go.Figure()
                ),
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q2 — Price by room type × neighbourhood:", className="query-label"),
                dcc.Dropdown(
                    id="q2-nbhd-dropdown",
                    options=[{"label": n, "value": n} for n in neighbourhoods],
                    value=neighbourhoods[:6] if len(neighbourhoods) >= 6 else neighbourhoods,
                    multi=True,
                    placeholder="Select neighbourhoods ..."
                ),
                dcc.Graph(id="q2-chart"),
            ]),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q4 — Price step-up per additional guest", className="query-label"),
                dcc.Graph(
                    figure=apply_theme(px.scatter(
                        q4, x="accommodates", y="avg_price", size="n_listings",
                        color_discrete_sequence=[AIRBNB_RED],
                        title="Q4 — Price vs Guest Capacity",
                        labels={"avg_price": "Avg Price (€)", "accommodates": "Guests"},
                    )) if not q4.empty else go.Figure()
                ),
            ], md=6),
            dbc.Col([
                html.P("Q5 — Availability tier vs price", className="query-label"),
                dcc.Graph(
                    figure=apply_theme(px.bar(
                        q5, x="availability_tier", y="avg_price",
                        color="availability_tier", color_discrete_sequence=AIRBNB_PALETTE,
                        title="Q5 — Avg Price by Availability Tier",
                        labels={"avg_price": "Avg Price (€)", "availability_tier": ""}
                    )) if not q5.empty else go.Figure()
                ),
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q6 — Multi-listing hosts vs single-listing hosts", className="query-label"),
                dcc.Graph(
                    figure=apply_theme(px.bar(
                        q6, x="host_portfolio", y="avg_price",
                        color="host_portfolio", color_discrete_sequence=AIRBNB_PALETTE,
                        title="Q6 — Avg Price by Host Portfolio Size",
                        labels={"avg_price": "Avg Price (€)", "host_portfolio": ""}
                    )) if not q6.empty else go.Figure()
                ),
            ], md=6),
            dbc.Col([
                html.P("Q7 — Price tier vs review scores", className="query-label"),
                dcc.Graph(
                    figure=apply_theme(px.imshow(
                        q7.set_index("price_tier")[["avg_overall", "avg_clean", "avg_location", "avg_value"]],
                        text_auto=".3f", color_continuous_scale="YlGnBu",
                        title="Q7 — Review Scores by Price Tier"
                    )) if not q7.empty else go.Figure()
                ),
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Q8 — Minimum-night policy distribution", className="query-label"),
                dcc.Graph(
                    figure=apply_theme(px.bar(
                        q8, x="min_night_bucket", y="n_listings",
                        color="min_night_bucket", text="pct_of_total",
                        color_discrete_sequence=AIRBNB_PALETTE,
                        title="Q8 — Minimum-Night Policy Buckets",
                        labels={"n_listings": "Listings", "min_night_bucket": ""}
                    )) if not q8.empty else go.Figure()
                ),
            ]),
        ], className="mb-4"),

        html.Hr(),
        html.H5("EDA Visualisations", className="sub-heading"),

        dbc.Row([
            dbc.Col([
                section_header("Room-Type Breakdown"),
                html.Label("Filter by neighbourhood:"),
                dcc.Dropdown(
                    id="eda-nbhd-dropdown",
                    options=[{"label": "All", "value": "ALL"}] + [{"label": n, "value": n} for n in neighbourhoods],
                    value="ALL",
                    clearable=False,
                ),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="eda-room-bar"), md=6),
                    dbc.Col(dcc.Graph(id="eda-room-box"), md=6),
                ]),
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_header("Monthly Review Volume — Demand Proxy"),
                dcc.Graph(id="review-ts-chart"),
                dcc.RangeSlider(
                    id="review-date-slider",
                    min=int(review_ts["date"].dt.year.min()) if not review_ts.empty else 2018,
                    max=int(review_ts["date"].dt.year.max()) if not review_ts.empty else 2024,
                    value=[2018, int(review_ts["date"].dt.year.max()) if not review_ts.empty else 2024],
                    marks={y: str(y) for y in range(2018, (int(review_ts["date"].dt.year.max()) if not review_ts.empty else 2024) + 1)},
                    step=1,
                ),
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_header("Superhost vs Non-Superhost"),
                dbc.Row([
                    dbc.Col(dcc.Graph(
                        figure=apply_theme(px.box(
                            superhost_comp, x="superhost_label", y="price",
                            color="superhost_label", color_discrete_sequence=[AIRBNB_TEAL, AIRBNB_RED],
                            title="Price: Superhost vs Non-Superhost",
                            labels={"price": "Nightly Price (€)", "superhost_label": ""},
                            points=False,
                        )) if not superhost_comp.empty else go.Figure()
                    ), md=6),
                    dbc.Col(dcc.Graph(
                        figure=apply_theme(px.box(
                            superhost_comp, x="superhost_label", y="review_scores_rating",
                            color="superhost_label", color_discrete_sequence=[AIRBNB_TEAL, AIRBNB_RED],
                            title="Rating: Superhost vs Non-Superhost",
                            labels={"review_scores_rating": "Review Score", "superhost_label": ""},
                            points=False,
                        )) if not superhost_comp.empty else go.Figure()
                    ), md=6),
                ]),
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_header("Minimum Nights Distribution"),
                html.Label("Cap histogram x-axis at (nights):"),
                dcc.Slider(10, 60, 10, value=60, id="minnights-cap-slider",
                           marks={i: str(i) for i in range(10, 61, 10)}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="minnights-hist"), md=6),
                    dbc.Col(dcc.Graph(
                        figure=apply_theme(
                            px.bar(
                                df[df["minimum_nights"] > 0].assign(
                                    min_stay_band=lambda x: pd.cut(
                                        x["minimum_nights"], bins=[0, 2, 7, 29, np.inf],
                                        labels=["1–2 nights", "3–7 nights", "8–29 nights", "30+ nights"]
                                    )
                                )["min_stay_band"].value_counts(sort=False)
                                .rename_axis("min_stay_band").reset_index(name="n_listings"),
                                x="min_stay_band", y="n_listings",
                                color="min_stay_band", color_discrete_sequence=AIRBNB_PALETTE,
                                title="Minimum-Stay Policy Buckets",
                                labels={"n_listings": "Listings", "min_stay_band": ""}
                            )
                        ) if "minimum_nights" in df.columns else go.Figure()
                    ), md=6),
                ]),
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_header("Host Portfolio Size"),
                html.Label("Cap histogram x-axis at (listings):"),
                dcc.Slider(5, 20, 5, value=20, id="portfolio-cap-slider",
                           marks={i: str(i) for i in range(5, 21, 5)}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="portfolio-hist"), md=6),
                    dbc.Col(dcc.Graph(
                        figure=apply_theme(
                            px.bar(
                                df[df["calculated_host_listings_count"] > 0].assign(
                                    host_type=lambda x: pd.cut(
                                        x["calculated_host_listings_count"], bins=[0, 1, 3, 10, np.inf],
                                        labels=["1 listing", "2–3 listings", "4–10 listings", "11+ listings"]
                                    )
                                )["host_type"].value_counts(sort=False)
                                .rename_axis("host_type").reset_index(name="n_listings"),
                                x="host_type", y="n_listings",
                                color="host_type", color_discrete_sequence=AIRBNB_PALETTE,
                                title="Single vs Multi-Listing Hosts",
                                labels={"n_listings": "Listings", "host_type": ""}
                            )
                        ) if "calculated_host_listings_count" in df.columns else go.Figure()
                    ), md=6),
                ]),
            ])
        ], className="mb-4"),

        html.Hr(),
        html.H5("Spatial Analysis", className="sub-heading"),

        dbc.Row([
            dbc.Col([
                map_card(
                    "Neighbourhood Price Choropleth",
                    choropleth_map_html,
                    "Could not load dublin_choropleth_price.html from /data."
                )
            ], md=12)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                map_card(
                    "Clickable Listings Map",
                    clickable_map_html,
                    "Could not load dublin_listings_clickable.html from /data."
                )
            ], md=12)
        ]),
    ], fluid=True, className="tab-content")

# =============================================================================
# TAB 2 — INTERACTIVE PRICE EXPLORER
# =============================================================================
def build_tab2():
    neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique()) if "neighbourhood_cleansed" in df.columns else []
    room_types = sorted(df["room_type"].dropna().unique()) if "room_type" in df.columns else []
    price_max = int(df["price"].quantile(0.99)) if "price" in df.columns else 1000

    return dbc.Container([
        section_header("Interactive Price Explorer", "Filter listings live by neighbourhood, room type, price and capacity"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Strong("🔍 Filters")),
                    dbc.CardBody([
                        html.Label("Neighbourhood"),
                        dcc.Dropdown(
                            id="exp-nbhd",
                            options=[{"label": n, "value": n} for n in neighbourhoods],
                            value=[],
                            multi=True,
                            placeholder="All neighbourhoods",
                        ),
                        html.Br(),
                        html.Label("Room Type"),
                        dcc.Checklist(
                            id="exp-room",
                            options=[{"label": rt, "value": rt} for rt in room_types],
                            value=room_types,
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"display": "block"},
                        ),
                        html.Br(),
                        html.Label("Price Range (€ / night)"),
                        dcc.RangeSlider(
                            id="exp-price",
                            min=15,
                            max=price_max,
                            value=[15, price_max],
                            step=5,
                            marks={15: "€15", price_max // 2: f"€{price_max // 2}", price_max: f"€{price_max}"},
                        ),
                        html.Br(),
                        html.Label("Guest Capacity"),
                        dcc.RangeSlider(
                            id="exp-acc",
                            min=1,
                            max=16,
                            value=[1, 16],
                            step=1,
                            marks={i: str(i) for i in range(1, 17, 3)},
                        ),
                        html.Br(),
                        html.Label("Host Type"),
                        dcc.RadioItems(
                            id="exp-superhost",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Superhost only", "value": "1"},
                                {"label": "Non-superhost only", "value": "0"},
                            ],
                            value="all",
                            labelStyle={"display": "block"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ])
                ], className="filter-card"),
            ], md=3),

            dbc.Col([
                dbc.Row(id="exp-kpis", className="mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="exp-price-hist"), md=6),
                    dbc.Col(dcc.Graph(id="exp-scatter"), md=6),
                ]),
                dbc.Row([dbc.Col(dcc.Graph(id="exp-nbhd-box"))], className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Filtered Listings", className="mt-3"),
                        dash_table.DataTable(
                            id="exp-table",
                            columns=[
                                {"name": "Neighbourhood", "id": "neighbourhood_cleansed"},
                                {"name": "Room Type", "id": "room_type"},
                                {"name": "Price (€)", "id": "price"},
                                {"name": "Accommodates", "id": "accommodates"},
                                {"name": "Bedrooms", "id": "bedrooms"},
                                {"name": "Rating", "id": "review_scores_rating"},
                                {"name": "Superhost", "id": "is_superhost"},
                            ],
                            page_size=20,
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": AIRBNB_RED, "color": "white", "fontWeight": "bold"},
                            style_cell={"fontFamily": "DM Sans, sans-serif", "padding": "6px"},
                            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
                        )
                    ])
                ], className="mt-3"),
            ], md=9),
        ]),
    ], fluid=True, className="tab-content")

# =============================================================================
# TAB 3 — STORYTELLING + REFLECTION
# =============================================================================
def build_tab3():
    sh_med = (
        superhost_comp.groupby("superhost_label")["price"].median()
        if not superhost_comp.empty else pd.Series(dtype=float)
    )

    return dbc.Container([
        section_header("Storytelling & Reflection", "Key findings, limitations, ethics, and future work"),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("🏡 What Makes a Dublin Airbnb Expensive?", className="story-h"),
                    dcc.Markdown("""
Dublin Airbnb prices are driven primarily by **listing size and room type**. Entire-home listings
sit at the top end of the market, while private and shared rooms form the budget segment.
Capacity, bedrooms, and bed count matter more than most host-related variables.

Neighbourhood also matters, but usually as a secondary effect. Once room type and size are
accounted for, location differences remain important but are less dramatic than they first appear.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("📊 Key Numbers at a Glance", className="story-h"),
                    dbc.Row([
                        dbc.Col(kpi_card("Median Price", f"€{df['price'].median():.0f}/night"), md=3),
                        dbc.Col(kpi_card("Superhost Median", f"€{sh_med.get('Superhost', 0):.0f}", color=AIRBNB_TEAL), md=3),
                        dbc.Col(kpi_card("Non-SH Median", f"€{sh_med.get('Non-superhost', 0):.0f}", color="#FC642D"), md=3),
                        dbc.Col(kpi_card("Neighbourhoods", str(df["neighbourhood_cleansed"].nunique()) if "neighbourhood_cleansed" in df.columns else "—", color="#FFB400"), md=3),
                    ]),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("🏅 The Superhost Pattern", className="story-h"),
                    dcc.Markdown(f"""
In this dataset, the median nightly price for a Superhost listing is **€{sh_med.get('Superhost', 0):.0f}**,
compared with **€{sh_med.get('Non-superhost', 0):.0f}** for non-Superhosts.

This suggests the Superhost badge may reflect **service quality and consistency**
more than luxury positioning alone.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("📈 Demand Signal: Monthly Reviews", className="story-h"),
                    dcc.Markdown("""
Monthly review counts can be used as a rough **demand proxy**. The time series shows:

- a sharp drop around the onset of COVID-19,
- gradual recovery through the following years,
- and visible seasonality, with stronger periods during peak travel months.

This makes the review series useful for explaining demand cycles, even if it is not a perfect
measure of bookings.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("⚠️ Data Limitations", className="story-h"),
                    dcc.Markdown("""
This dataset is a **snapshot of listed properties**, not actual transaction-level booking data.
Listed prices may differ from what guests really pay after discounts, cleaning fees, or dynamic pricing changes.

Review scores are also imperfect because many dissatisfied guests do not leave reviews, which compresses
the lower end of the rating scale.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("🔬 Analytical Limitations", className="story-h"),
                    dcc.Markdown("""
The dashboard focuses on descriptive and exploratory analysis. It does not fully model spatial dependence,
seasonal booking dynamics, or causal relationships between listing features and price.

In addition, some variables such as review scores may be correlated with price in ways that are difficult
to interpret cleanly.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("⚖️ Ethical Considerations", className="story-h"),
                    dcc.Markdown("""
Short-term rental platforms may affect long-term housing supply in cities such as Dublin.
Tools that help hosts optimise prices can improve efficiency, but they may also contribute to affordability pressures.

This means dashboard outputs should be interpreted as **descriptive insights**, not normative recommendations.
                    """, className="story-body"),
                ]), className="story-card mb-3"),

                dbc.Card(dbc.CardBody([
                    html.H5("🚀 Future Work", className="story-h"),
                    dcc.Markdown("""
Future improvements could include:

1. richer spatial comparisons across neighbourhoods,
2. transaction-level or booking-level price data,
3. natural-language analysis of listing descriptions,
4. and stronger time-series modelling for seasonality.
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
server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Span("airbnb", className="logo-text"),
            html.Span(" analytics", className="logo-sub"),
        ], className="logo"),
        html.Div([
            html.H1("Dublin Airbnb Dashboard", className="header-title"),
            html.P("Data: Inside Airbnb · Dublin · Group 7", className="header-sub"),
        ], className="header-text"),
    ], className="dashboard-header"),

    dcc.Tabs(id="main-tabs", value="tab1", className="main-tabs", children=[
        dcc.Tab(label="① Data & EDA", value="tab1", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="② Price Explorer", value="tab2", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="③ Storytelling & Reflection", value="tab3", className="tab", selected_className="tab--selected"),
    ]),
    html.Div(id="tab-content"),
], className="app-shell")

# =============================================================================
# CALLBACKS
# =============================================================================
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab1":
        return build_tab1()
    if tab == "tab2":
        return build_tab2()
    if tab == "tab3":
        return build_tab3()
    return build_tab1()

@app.callback(Output("missing-bar", "figure"), Input("missing-n-slider", "value"))
def update_missing_bar(n):
    data = missing_report.head(n)
    if data.empty:
        return go.Figure()
    fig = px.bar(
        data, x="pct", y="column", orientation="h",
        color="pct",
        color_continuous_scale=[[0, "#FFB400"], [1, AIRBNB_RED]],
        title=f"Top {n} Columns by % Missing",
        labels={"pct": "% Missing", "column": ""}
    )
    return apply_theme(fig)

@app.callback(Output("q2-chart", "figure"), Input("q2-nbhd-dropdown", "value"))
def update_q2(selected):
    if q2.empty:
        return go.Figure()
    data = q2 if not selected else q2[q2["neighbourhood"].isin(selected)]
    fig = px.bar(
        data, x="neighbourhood", y="avg_price", color="room_type",
        barmode="group", color_discrete_sequence=AIRBNB_PALETTE,
        title="Q2 — Avg Price by Room Type × Neighbourhood",
        labels={"avg_price": "Avg Price (€)", "neighbourhood": ""}
    )
    return apply_theme(fig)

@app.callback(
    Output("eda-room-bar", "figure"),
    Output("eda-room-box", "figure"),
    Input("eda-nbhd-dropdown", "value"),
)
def update_eda_room(nbhd):
    if "room_type" not in df.columns:
        return go.Figure(), go.Figure()

    subset = df if nbhd == "ALL" else df[df["neighbourhood_cleansed"] == nbhd]
    rc = (
        subset["room_type"]
        .value_counts()
        .rename_axis("room_type")
        .reset_index(name="count")
    )

    fig_bar = apply_theme(px.bar(
        rc, x="room_type", y="count", color="room_type",
        color_discrete_sequence=AIRBNB_PALETTE,
        title="Listings by Room Type",
        labels={"count": "Listings", "room_type": ""}
    ))

    fig_box = apply_theme(px.box(
        subset, x="room_type", y="price", color="room_type",
        color_discrete_sequence=AIRBNB_PALETTE,
        title="Price by Room Type",
        labels={"price": "Nightly Price (€)", "room_type": ""},
        points=False
    ))

    return fig_bar, fig_box

@app.callback(Output("review-ts-chart", "figure"), Input("review-date-slider", "value"))
def update_review_ts(years):
    if review_ts.empty:
        return go.Figure()

    data = review_ts[
        (review_ts["date"].dt.year >= years[0]) &
        (review_ts["date"].dt.year <= years[1])
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["review_count"],
        fill="tozeroy", line=dict(color=AIRBNB_RED, width=2),
        name="Reviews / Month"
    ))
    fig.add_vrect(
        x0="2020-03-01", x1="2021-06-01",
        fillcolor="gray", opacity=0.12,
        annotation_text="COVID-19 restrictions", line_width=0
    )
    fig.update_layout(title="Monthly Review Volume (demand proxy)", **PLOT_THEME)
    return fig

@app.callback(Output("minnights-hist", "figure"), Input("minnights-cap-slider", "value"))
def update_minnights(cap):
    if "minimum_nights" not in df.columns:
        return go.Figure()
    data = df[(df["minimum_nights"] > 0) & (df["minimum_nights"] <= cap)]
    fig = px.histogram(
        data, x="minimum_nights", nbins=cap,
        color_discrete_sequence=[AIRBNB_RED],
        title=f"Min Nights Distribution (cap={cap})",
        labels={"minimum_nights": "Minimum Nights"}
    )
    fig.add_vline(x=30, line_dash="dash", line_color="#767676", annotation_text="30-night threshold")
    return apply_theme(fig)

@app.callback(Output("portfolio-hist", "figure"), Input("portfolio-cap-slider", "value"))
def update_portfolio(cap):
    if "calculated_host_listings_count" not in df.columns:
        return go.Figure()
    data = df[
        (df["calculated_host_listings_count"] > 0) &
        (df["calculated_host_listings_count"] <= cap)
    ]
    return apply_theme(px.histogram(
        data, x="calculated_host_listings_count", nbins=cap,
        color_discrete_sequence=[AIRBNB_TEAL],
        title=f"Listings per Host (cap={cap})",
        labels={"calculated_host_listings_count": "Listings per Host"}
    ))

@app.callback(
    Output("exp-kpis", "children"),
    Output("exp-price-hist", "figure"),
    Output("exp-scatter", "figure"),
    Output("exp-nbhd-box", "figure"),
    Output("exp-table", "data"),
    Input("exp-nbhd", "value"),
    Input("exp-room", "value"),
    Input("exp-price", "value"),
    Input("exp-acc", "value"),
    Input("exp-superhost", "value"),
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
        dbc.Col(kpi_card("Listings", f"{len(sub):,}"), md=3),
        dbc.Col(kpi_card("Median Price", f"€{sub['price'].median():.0f}" if len(sub) else "—", color=AIRBNB_TEAL), md=3),
        dbc.Col(kpi_card("Avg Rating", f"{sub['review_scores_rating'].mean():.2f}" if len(sub) else "—", color="#FC642D"), md=3),
        dbc.Col(kpi_card("Avg Capacity", f"{sub['accommodates'].mean():.1f} guests" if len(sub) else "—", color="#FFB400"), md=3),
    ])

    if len(sub) == 0:
        empty = go.Figure()
        return kpis, empty, empty, empty, []

    fig_hist = apply_theme(px.histogram(
        sub, x="price", nbins=60,
        color_discrete_sequence=[AIRBNB_RED],
        title="Price Distribution (filtered)",
        labels={"price": "Nightly Price (€)"}
    ))

    fig_scat = apply_theme(px.scatter(
        sub.sample(min(1000, len(sub)), random_state=1),
        x="accommodates", y="price", color="room_type",
        color_discrete_sequence=AIRBNB_PALETTE, opacity=0.6,
        title="Price vs Capacity",
        labels={"price": "Nightly Price (€)", "accommodates": "Guests"}
    ))

    top_n = sub["neighbourhood_cleansed"].value_counts().head(15).index.tolist()
    sub_top = sub[sub["neighbourhood_cleansed"].isin(top_n)]

    fig_box = apply_theme(px.box(
        sub_top, x="neighbourhood_cleansed", y="price",
        color="neighbourhood_cleansed",
        color_discrete_sequence=AIRBNB_PALETTE,
        title="Price by Neighbourhood (top 15)",
        labels={"price": "Nightly Price (€)", "neighbourhood_cleansed": ""},
        points=False
    ))
    fig_box.update_layout(showlegend=False)

    tbl_cols = [
        "neighbourhood_cleansed", "room_type", "price",
        "accommodates", "bedrooms", "review_scores_rating", "is_superhost"
    ]
    tbl_data = sub[[c for c in tbl_cols if c in sub.columns]].head(200).round(2).to_dict("records")

    return kpis, fig_hist, fig_scat, fig_box, tbl_data

# =============================================================================
# INLINE CSS
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
  .main-tabs { background: var(--dark); padding: 0 24px; border: none !important; }
  .tab { background: transparent !important; border: none !important;
         color: #bbb !important; padding: 14px 18px !important;
         font-size: 13px; font-weight: 500; letter-spacing: 0.3px; }
  .tab:hover { color: white !important; }
  .tab--selected { color: white !important; border-bottom: 3px solid var(--red) !important;
                   background: rgba(255,90,95,0.12) !important; }
  .tab-content { padding: 28px 24px; }
  .section-header { margin-bottom: 16px; }
  .section-title  { font-family: 'DM Serif Display', serif; font-size: 20px;
                    color: var(--dark); margin-bottom: 2px; }
  .section-sub    { font-size: 12px; color: var(--grey); margin-bottom: 6px; }
  .section-rule   { border: none; border-top: 2px solid var(--red);
                    opacity: 0.3; margin: 6px 0 12px; }
  .sub-heading    { font-size: 16px; font-weight: 600; color: var(--dark);
                    margin: 20px 0 10px; padding-left: 10px;
                    border-left: 4px solid var(--red); }
  .kpi-card   { border: none; border-radius: var(--radius); background: var(--white);
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 100%; }
  .kpi-label  { font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                color: var(--grey); margin-bottom: 4px; }
  .kpi-value  { font-size: 24px; font-weight: 700; margin: 0; }
  .kpi-sub    { font-size: 11px; color: var(--grey); margin-top: 2px; }
  .query-label { font-size: 12px; font-style: italic; color: var(--grey); margin-bottom: 4px; }
  .story-card { border: none; border-radius: var(--radius); background: var(--white);
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .story-h    { font-family: 'DM Serif Display', serif; font-size: 18px; color: var(--dark); }
  .story-body { font-size: 14px; line-height: 1.7; color: #444; }
  .filter-card { border: none; border-radius: var(--radius); background: var(--white);
                 box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 100%; }
  .filter-card .card-header { background: var(--dark); color: white;
                               border-radius: var(--radius) var(--radius) 0 0; }
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
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050))
    )
