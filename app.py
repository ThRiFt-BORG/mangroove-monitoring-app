# Dash is our main app library
import dash
from dash import dcc, html, Input, Output

# Data handling
import pandas as pd
import numpy as np

# Raster/image handling
import rasterio
import matplotlib.pyplot as plt
import os
import base64

# File locations
AREA_CSV = "data/changes/mangrove_area_change.csv"
NDVI_CSV = "data/changes/ndvi_ndmi_stats.csv"

MASK_DIR = "data/masks"
NDVI_DIR = "data/ndvi"
NDMI_DIR = "data/ndmi"
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
    with rasterio.open(tif_path) as src:
        array = src.read(1)

    cmap = {
        "ndvi": "Greens",
        "ndmi": "Blues",
        "mask": "gray"
    }.get(label, "viridis")

    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(array, cmap=cmap)
    ax.axis("off")

    if label != "mask":
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, orientation="vertical")
        cbar.set_label("Index Value", fontsize=8)

    fname = f"{EXPORT_DIR}/{label}_{year}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=100)
    plt.close()

    with open(fname, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# === Callbacks ===
@app.callback(
    [Output("ndvi-ndmi-graph", "figure"),
     Output("area-graph", "figure"),
     Output("ndvi-img", "src"),
     Output("ndmi-img", "src"),
     Output("mask-img", "src")],
    [Input("year-dropdown", "value")]
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
