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

# =============================================================================
# BRAND STYLE
# =============================================================================

AIRBNB_PALETTE = ["#FF5A5F","#00A699","#FC642D","#484848","#767676","#FFB400"]

AIRBNB_RED="#FF5A5F"
AIRBNB_TEAL="#00A699"

CITY_NAME="Dublin"

# =============================================================================
# DATA
# =============================================================================

DATA_DIR="data"

LISTINGS_PATH=os.path.join(DATA_DIR,"listings.csv.gz")
REVIEWS_PATH=os.path.join(DATA_DIR,"reviews.csv")

listings_raw=pd.read_csv(
    LISTINGS_PATH,
    low_memory=False,
    compression="gzip"
)

reviews_raw=pd.read_csv(
    REVIEWS_PATH,
    low_memory=False
)

# =============================================================================
# CLEANING
# =============================================================================

def parse_price(df):

    df["price"]=(
        df["price"]
        .astype(str)
        .str.replace(r"[\$,€]","",regex=True)
        .astype(float)
    )

    return df


df=(
    listings_raw
    .pipe(parse_price)
)

df=df.dropna(
    subset=["price","latitude","longitude"]
)

df=df.query("15<=price<=1000")

df["log_price"]=np.log1p(df["price"])

df["is_superhost"]=(
    df["host_is_superhost"]
    .map({"t":1,"f":0})
)

# =============================================================================
# SQL
# =============================================================================

sql=lambda q:sqldf(q,{"df":df})

q1=sql("""

SELECT

neighbourhood_cleansed as neighbourhood,

COUNT(*) listings,

AVG(price) avg_price

FROM df

GROUP BY neighbourhood_cleansed

ORDER BY avg_price DESC

""")

# =============================================================================
# DASH APP
# =============================================================================

app=Dash(

__name__,

external_stylesheets=[

dbc.themes.BOOTSTRAP

],

suppress_callback_exceptions=True

)

server=app.server

# =============================================================================
# HELPERS
# =============================================================================

def section_header(title,subtitle=""):

    return html.Div([

        html.H4(title),

        html.P(subtitle),

        html.Hr()

    ])


def kpi(title,value):

    return dbc.Card(

        dbc.CardBody([

            html.P(title),

            html.H3(value)

        ])

    )

# =============================================================================
# TAB 1
# DATA + EDA
# =============================================================================

def build_tab1():

    fig_price=px.histogram(

        df,

        x="price",

        nbins=80,

        title="Price Distribution"

    )

    fig_room=px.box(

        df,

        x="room_type",

        y="price",

        title="Price by Room Type"

    )

    return dbc.Container([

        section_header(

        "Data Wrangling & EDA",

        "Cleaning + exploration"

        ),

        dbc.Row([

        dbc.Col(kpi(

        "Listings",

        len(df)

        )),

        dbc.Col(kpi(

        "Median Price",

        f"€{df.price.median():.0f}"

        )),

        dbc.Col(kpi(

        "Neighbourhoods",

        df.neighbourhood_cleansed.nunique()

        ))

        ]),

        html.Br(),

        dcc.Graph(

        figure=fig_price

        ),

        dcc.Graph(

        figure=fig_room

        ),

        html.H5(

        "Neighbourhood price ranking"

        ),

        dash_table.DataTable(

        data=q1.to_dict("records"),

        columns=[

        {"name":i,"id":i}

        for i in q1.columns

        ],

        page_size=15

        )

    ])

# =============================================================================
# TAB 2
# EXPLORER
# =============================================================================

def build_tab2():

    return dbc.Container([

    section_header(

    "Price Explorer"

    ),

    dcc.Graph(

    figure=px.scatter(

    df,

    x="accommodates",

    y="price",

    color="room_type"

    )

    )

    ])

# =============================================================================
# TAB 3
# MAPS
# =============================================================================

def build_tab3():

    choropleth=open(

    "data/dublin_choropleth_price.html"

    ).read()

    listings_map=open(

    "data/dublin_listings_clickable.html"

    ).read()

    return dbc.Container([

    section_header(

    "Spatial Analysis",

    "Geographic view"

    ),

    html.H5(

    "Neighbourhood Prices"

    ),

    html.Iframe(

    srcDoc=choropleth,

    style={

    "width":"100%",

    "height":"600px",

    "border":"none"

    }

    ),

    html.Br(),

    html.H5(

    "Listings Map"

    ),

    html.Iframe(

    srcDoc=listings_map,

    style={

    "width":"100%",

    "height":"600px",

    "border":"none"

    }

    )

    ])

# =============================================================================
# TAB 4
# STORY + REFLECTION
# =============================================================================

def build_tab4():

    return dbc.Container([

    section_header(

    "Insights & Reflection"

    ),

    dbc.Card(

    dbc.CardBody([

    html.H5(

    "Key Findings"

    ),

    dcc.Markdown("""

Entire homes dominate high prices.

Capacity strongly drives price.

Superhosts slightly higher priced.

    """)

    ])

    ),

    html.Br(),

    dbc.Card(

    dbc.CardBody([

    html.H5(

    "Limitations"

    ),

    dcc.Markdown("""

Listing prices ≠ transaction prices.

Snapshot data.

No causal claims.

    """)

    ])

    )

    ])

# =============================================================================
# LAYOUT
# =============================================================================

app.layout=html.Div([

html.H1(

"Dublin Airbnb Dashboard"

),

dcc.Tabs(

id="tabs",

value="tab1",

children=[

dcc.Tab(

label="① Data & EDA",

value="tab1"

),

dcc.Tab(

label="② Explorer",

value="tab2"

),

dcc.Tab(

label="③ Maps",

value="tab3"

),

dcc.Tab(

label="④ Insights",

value="tab4"

)

]

),

html.Div(

id="content"

)

])

# =============================================================================
# CALLBACK
# =============================================================================

@app.callback(

Output("content","children"),

Input("tabs","value")

)

def render(tab):

    if tab=="tab1":

        return build_tab1()

    if tab=="tab2":

        return build_tab2()

    if tab=="tab3":

        return build_tab3()

    if tab=="tab4":

        return build_tab4()

    return build_tab1()

# =============================================================================
# RUN
# =============================================================================

if __name__=="__main__":

    app.run_server(debug=True)
