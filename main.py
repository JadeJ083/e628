from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("Dublin Airbnb Dashboard", className="mt-4"),
    html.P("App deployed successfully.")
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
