import os
import glob
import dash
import rasterio
import numpy as np
import pandas as pd
import traceback

# Set Matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm

import base64
import folium
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# === File Locations ===
NDVI_CSV = "data/changes/ndvi_ndmi_stats.csv"
MASK_DIR = "data/masks"
NDVI_DIR = "data/ndvi"
NDMI_DIR = "data/ndmi"
CHANGES_DIR = "data/changes"
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Load CSVs
area_df = pd.read_csv(AREA_CSV)
ndvi_df = pd.read_csv(NDVI_CSV)
YEARS = sorted(area_df["Year"].astype(str).tolist())

# Initialize app
app = dash.Dash(__name__)
app.title = "Mangrove Monitoring Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Mangrove Monitoring Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id="year-dropdown",
            options=[{"label": y, "value": y} for y in YEARS],
            value=YEARS[0],
            style={"width": "100%"}
        )
    ], style={"width": "30%", "margin": "auto"}),

    html.Br(),

    # === SIDE-BY-SIDE GRAPHS ===
    html.Div([
        dcc.Graph(id="ndvi-ndmi-graph", style={"width": "50%"}),
        dcc.Graph(id="area-graph", style={"width": "50%"})
    ], style={"display": "flex", "justifyContent": "space-around"}),

    html.Br(),

    # === IMAGE FRAMES ===
    html.Div([
        html.H3("Raster's Preview", style={"textAlign": "center"}),

        html.Div([
            html.Div([
                html.H5("NDVI"),
                html.Img(id="ndvi-img", style={"width": "100%", "border": "1px solid #ccc"})
            ], style={"width": "32%", "textAlign": "center"}),

            html.Div([
                html.H5("NDMI"),
                html.Img(id="ndmi-img", style={"width": "100%", "border": "1px solid #ccc"})
            ], style={"width": "32%", "textAlign": "center"}),

            html.Div([
                html.H5("Mangrove Mask"),
                html.Img(id="mask-img", style={"width": "100%", "border": "1px solid #ccc"})
            ], style={"width": "32%", "textAlign": "center"})
        ], style={"display": "flex", "justifyContent": "space-around", "padding": "0 20px"})
    ])
])


# === Image rendering helper ===
def render_tif_to_png(tif_path, label, year):
    if not os.path.exists(tif_path): return ""
    with rasterio.open(tif_path) as src: array = src.read(1)
    
   
    # The first color, '#00000000', is transparent for background (value 0).
    cmap_spec = {
        "ndvi": "Greens", 
        "ndmi": "Blues", 
        "classified": ["#0000FF", "#FFA500", "#008000", "#964B00"], 
        "gain": ['#00000000', 'lime'],  
        "loss": ['#00000000', 'red']   
    }.get(label.split('_')[0], "viridis")
    
    # Check if the spec is a name (string) or a list for a custom map
    if isinstance(cmap_spec, str):
        cmap = cmap_spec
    else:
        cmap = plt.cm.colors.ListedColormap(cmap_spec)
        
    fig, ax = plt.subplots(figsize=(3, 3)); ax.axis("off")
    im = ax.imshow(array, cmap=cmap)
    
    if label in ["ndvi", "ndmi"]: fig.colorbar(im, ax=ax, shrink=0.75, orientation="vertical").set_label("Index Value", fontsize=8)
    
    fname = f"{EXPORT_DIR}/{label}_{year}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=100); plt.close(fig)
    
    with open(fname, "rb") as f: return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# === Initialize App & Layouts  ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
# ... rest of the layout and callback code is identical to your last version ...
app.title = "Mangrove Monitoring Dashboard"
app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    html.H1("Mangrove Monitoring Dashboard", className="text-center my-4"),
    dbc.Nav([dbc.NavLink("Report", href="/report", active="exact"), dbc.NavLink("Results", href="/results", active="exact")]),
    html.Div(id="page-content")
], fluid=True)

report_layout = html.Div([
    html.H2("Mangrove Monitoring Report", className="text-center my-4"),
    dcc.Dropdown(id="year-select", options=[{"label": y, "value": y} for y in range(2018, 2026)], value=2025, className="my-4"),
    dbc.Row([dbc.Col(dcc.Graph(id="areas-graph"), width=6), dbc.Col(dcc.Graph(id="index-stats-graph"), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="metrics-graph"), width=6), dbc.Col(dcc.Graph(id="band-importance-graph"), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id="report-gain-loss-graph"), width=6)])
])

results_layout = html.Div([
    html.H2("Mangrove Monitoring Results", className="text-center my-4"),
    dcc.Slider(id="year-slider", min=2019, max=2025, step=1, value=2025, marks={y: str(y) for y in range(2019, 2026)}, className="my-4"),
    dcc.Dropdown(id="layer-select", options=[
        {"label": "NDVI", "value": "ndvi"}, {"label": "NDMI", "value": "ndmi"}, {"label": "Classified", "value": "classified"},
        {"label": "Gain (NDVI ≥ 0.3)", "value": "gain_0.3"}, {"label": "Loss (NDVI ≥ 0.3)", "value": "loss_0.3"},
        {"label": "Gain (NDVI ≥ 0.4)", "value": "gain_0.4"}, {"label": "Loss (NDVI ≥ 0.4)", "value": "loss_0.4"},
        {"label": "Gain (NDVI ≥ 0.5)", "value": "gain_0.5"}, {"label": "Loss (NDVI ≥ 0.5)", "value": "loss_0.5"}
    ], value="classified", className="my-4"),

    # === 2x2 GRID STRUCTURE ===

    # --- TOP ROW: Map and Raster Preview ---
    dbc.Row([
        # Column 1: The Map
        dbc.Col([
            html.H4("Map"), 
            # The Iframe for the Folium map can have its height set directly
            html.Iframe(id="map", style={'height': '500px', 'width': '100%'})
        ], width=6), # width=6 means this column takes up half the screen

        # Column 2: The Raster Preview Image
        dbc.Col([
            html.H4("Raster Preview"), 
            # Center the image and ensure it doesn't overflow
            html.Div(
                html.Img(id="raster-preview", style={"max-width": "100%", "max-height": "450px", "border": "1px solid #ccc"}),
                className="d-flex justify-content-center align-items-center" # Bootstrap classes for centering
            )
        ], width=6) # width=6 means this column also takes up half the screen
    ], className="mb-4"), # mb-4 adds a margin to the bottom of the row

    # --- BOTTOM ROW: The two graphs ---
    dbc.Row([
        # Column 1: Area Trends Graph
        dbc.Col([
            html.H4("Class Area Trends Over Time"), 
            dcc.Graph(id="area-trend-graph", style={'height': '450px'})
        ], width=6),

        # Column 2: Gain/Loss Graph
        dbc.Col([
            html.H4("Gain/Loss Analysis"), 
            dcc.Graph(id="gain-loss-graph", style={'height': '450px'})
        ], width=6)
    ])
])
# === Callbacks (No changes from previous version) ===
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    return results_layout if pathname == "/results" else report_layout

@app.callback(
    [Output("areas-graph", "figure"), Output("index-stats-graph", "figure"), Output("metrics-graph", "figure"),
     Output("band-importance-graph", "figure"), Output("report-gain-loss-graph", "figure")],
    Input("year-select", "value")
)
def update_report_page_graphs(year):
    if year is None: return [{"data": [], "layout": {"title": "No Data Available"}}] * 5
    empty_fig = {"data": [], "layout": {"title": "No Data Available"}}
    areas_fig, index_stats_fig, metrics_fig, band_fig, gain_loss_fig = [empty_fig] * 5
    areas_file = os.path.join(CHANGES_DIR, f"Classified_Areas_{year}.csv")
    metrics_file = os.path.join(CHANGES_DIR, f"Classification_Metrics_{year}.csv")
    band_file = os.path.join(CHANGES_DIR, f"Band_Importance_{year}.csv")
    gain_loss_file = os.path.join(CHANGES_DIR, f"Change_Detection_{year-1}_to_{year}_NDVI.csv")
    ndvi_stats_file = NDVI_CSV
    if os.path.exists(areas_file):
        try:
            desired_cols = ['Year', 'Water', 'Bareland', 'Mangrove', 'Prosopis']
            areas_df = pd.read_csv(areas_file, usecols=lambda c: c in desired_cols).fillna(0)
            value_vars = [col for col in desired_cols if col in areas_df.columns and col != 'Year']
            areas_df_long = areas_df.melt(id_vars="Year", value_vars=value_vars, var_name="Class", value_name="Area (m²)")
            areas_fig = px.bar(areas_df_long, x="Class", y="Area (m²)", title=f"Classified Areas for {year}", color="Class",
                               color_discrete_map={"Water": "#0000FF", "Bareland": "#FFA500", "Mangrove": "#008000", "Prosopis": "#964B00"})
        except Exception as e: print(f"Error processing {areas_file}: {e}")
    if os.path.exists(ndvi_stats_file):
        year_stats = pd.read_csv(ndvi_stats_file).query(f"Year == {year}").melt(id_vars="Year", var_name="Statistic", value_name="Value")
        index_stats_fig = px.bar(year_stats, x="Statistic", y="Value", title=f"NDVI/NDMI Statistics for {year}", color="Statistic")
    if os.path.exists(metrics_file):
        try:
            desired_cols = ['Year', 'Validation_Accuracy', 'Mangrove_Precision', 'Mangrove_Recall']
            metrics_df = pd.read_csv(metrics_file, usecols=lambda c: c in desired_cols).fillna(0)
            metrics_fig = px.bar(metrics_df.melt(id_vars="Year", var_name="Metric", value_name="Score"), x="Metric", y="Score", title=f"Classification Metrics for {year}", range_y=[0,1])
        except Exception as e: print(f"Error processing {metrics_file}: {e}")
    if os.path.exists(band_file):
        try:
            desired_cols = ["Year", "B3", "B4", "B6", "B8", "Entropy", "NDMI", "NDVI", "NDWI"]
            band_df = pd.read_csv(band_file, usecols=lambda c: c in desired_cols).fillna(0)
            band_fig = px.bar(band_df.melt(id_vars="Year", var_name="Band", value_name="Importance"), x="Band", y="Importance", title=f"Band Importance for {year}")
        except Exception as e: print(f"Error processing {band_file}: {e}")
    if os.path.exists(gain_loss_file):
        gain_loss_df = pd.read_csv(gain_loss_file)
        gain_loss_fig = px.bar(gain_loss_df, x="NDVI_Threshold", y=["Gain_m2", "Loss_m2"], title=f"Gain & Loss from {year-1} to {year}", barmode='group')
    return areas_fig, index_stats_fig, metrics_fig, band_fig, gain_loss_fig

@app.callback(
    [Output("map", "srcDoc"), Output("raster-preview", "src"), Output("area-trend-graph", "figure"), Output("gain-loss-graph", "figure")],
    [Input("year-slider", "value"), Input("layer-select", "value")]
)
def update_dashboard(selected_year):
    year = int(selected_year)

    # NDVI/NDMI line chart
    ndvi_trace = dict(x=ndvi_df["Year"], y=ndvi_df["Ndvi_Mean"],
                      type="line", name="NDVI Mean", line=dict(color="green"))
    ndmi_trace = dict(x=ndvi_df["Year"], y=ndvi_df["Ndmi_Mean"],
                      type="line", name="NDMI Mean", line=dict(color="blue"))
    ndvi_ndmi_fig = {
        "data": [ndvi_trace, ndmi_trace],
        "layout": {
            "title": "NDVI & NDMI Mean Trends",
            "xaxis": {"title": "Year"},
            "yaxis": {"title": "Mean Index Value"},
        }
    }

    # Area bar chart (m²)
    area_fig = {
        "data": [
            dict(x=area_df["Year"], y=area_df["Area_m2"], type="bar",
                 name="Mangrove Area (m²)", marker=dict(color="seagreen"))
        ],
        "layout": {
            "title": "Mangrove Area (in m²)",
            "xaxis": {"title": "Year"},
            "yaxis": {"title": "Area (m²)"}
        }
    }

    # Render images for selected year
    ndvi_img = render_tif_to_png(os.path.join(NDVI_DIR, f"ndvi_{year}.tif"), "ndvi", year)
    ndmi_img = render_tif_to_png(os.path.join(NDMI_DIR, f"ndmi_{year}.tif"), "ndmi", year)
    mask_img = render_tif_to_png(os.path.join(MASK_DIR, f"mask_{year}.tif"), "mask", year)

    return ndvi_ndmi_fig, area_fig, ndvi_img, ndmi_img, mask_img


# Run app
if __name__ == "__main__":
    app.run(debug=True)
