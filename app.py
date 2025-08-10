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

# === Precompute Calculations  ===
def compute_gain_loss(year, thresholds=[0.3, 0.4, 0.5], data_dir="data"):
    ndvi_dir = os.path.join(data_dir, "ndvi")
    masks_dir = os.path.join(data_dir, "masks")
    changes_dir = os.path.join(data_dir, "changes")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(changes_dir, exist_ok=True)
    curr_file = os.path.join(ndvi_dir, f"ndvi_{year}.tif")
    prev_file = os.path.join(ndvi_dir, f"ndvi_{year-1}.tif")
    if not (os.path.exists(curr_file) and os.path.exists(prev_file)):
        return pd.DataFrame()
    with rasterio.open(curr_file) as curr_ds, rasterio.open(prev_file) as prev_ds:
        curr_ndvi, prev_ndvi = curr_ds.read(1), prev_ds.read(1)
        profile, transform = curr_ds.profile, curr_ds.transform
        pixel_area = abs(transform[0] * transform[4])
        results = []
        for th in thresholds:
            gain = (curr_ndvi >= th) & ~(prev_ndvi >= th)
            loss = (prev_ndvi >= th) & ~(curr_ndvi >= th)
            results.append({
                "Year": year, "Previous_Year": year - 1, "NDVI_Threshold": th,
                "Gain_m2": np.sum(gain) * pixel_area, "Loss_m2": np.sum(loss) * pixel_area,
                "Net_m2": (np.sum(gain) - np.sum(loss)) * pixel_area
            })
            for mask, name in [(gain, "Gain"), (loss, "Loss")]:
                output_file = os.path.join(masks_dir, f"{name}_NDVI_{th}_{year-1}_to_{year}.tif")
                with rasterio.open(output_file, "w", **profile) as dst:
                    dst.write(mask.astype(np.uint8)[np.newaxis, :, :])
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(changes_dir, f"Change_Detection_{year-1}_to_{year}_NDVI.csv"), index=False)
        return df

def compute_index_stats(data_dir="data"):
    results = []
    for year in range(2018, 2026):
        ndvi_file = os.path.join(data_dir, "ndvi", f"ndvi_{year}.tif")
        ndmi_file = os.path.join(data_dir, "ndmi", f"ndmi_{year}.tif")
        if os.path.exists(ndvi_file) and os.path.exists(ndmi_file):
            with rasterio.open(ndvi_file) as ndvi_ds, rasterio.open(ndmi_file) as ndmi_ds:
                results.append({
                    "Year": year, "Ndvi_Mean": np.nanmean(ndvi_ds.read(1)), "Ndvi_Std": np.nanstd(ndvi_ds.read(1)),
                    "Ndmi_Mean": np.nanmean(ndmi_ds.read(1)), "Ndmi_Std": np.nanstd(ndmi_ds.read(1))
                })
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(NDVI_CSV, index=False)
    return df

for year in range(2019, 2026):
    compute_gain_loss(year)
compute_index_stats()

# === TIFF RENDERING
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
def update_results_page(year, layer):
    empty_fig = {"data": [], "layout": {"title": "Error Loading Data"}}
    error_return = "", "", empty_fig, empty_fig
    try:
        if year is None or layer is None: return "", "", {"data": [], "layout": {"title": "No Data Available"}}, {"data": [], "layout": {"title": "No Data Available"}}
        map_html, preview_src = "", ""
        tif_path = None
        if layer == "ndvi": tif_path = os.path.join(NDVI_DIR, f"ndvi_{year}.tif")
        elif layer == "ndmi": tif_path = os.path.join(NDMI_DIR, f"ndmi_{year}.tif")
        elif layer == "classified": tif_path = os.path.join(MASK_DIR, f"Classified_{year}.tif")
        elif layer.startswith("gain"): tif_path = os.path.join(MASK_DIR, f"Gain_NDVI_{layer.split('_')[1]}_{year-1}_to_{year}.tif")
        elif layer.startswith("loss"): tif_path = os.path.join(MASK_DIR, f"Loss_NDVI_{layer.split('_')[1]}_{year-1}_to_{year}.tif")
        
        if tif_path and os.path.exists(tif_path):
            with rasterio.open(tif_path) as src:
                raster_array, bounds = src.read(1), src.bounds
                m = folium.Map(location=[(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2], zoom_start=10)
                m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
                cmap = {"ndvi": plt.cm.Greens, "ndmi": plt.cm.Blues, "classified": plt.cm.colors.ListedColormap(["#0000FF", "#FFA500", "#008000", "#964B00"]),
                        "gain": plt.cm.colors.ListedColormap(['#00000000', 'lime']), "loss": plt.cm.colors.ListedColormap(['#00000000', 'red'])}.get(layer.split('_')[0])
                folium.raster_layers.ImageOverlay(image=raster_array, bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]], opacity=0.7, colormap=cmap).add_to(m)
                map_html = m._repr_html_()
            preview_src = render_tif_to_png(tif_path, layer, year)

        area_fig = {"data": [], "layout": {"title": "No Area Data Available"}}
        all_areas_files = glob.glob(os.path.join(CHANGES_DIR, "Classified_Areas_*.csv"))
        if all_areas_files:
            desired_cols = ['Year', 'Water', 'Bareland', 'Mangrove', 'Prosopis']
            df_list = [pd.read_csv(f, usecols=lambda c: c in desired_cols) for f in all_areas_files if os.path.exists(f)]
            if df_list:
                areas_trend_df = pd.concat(df_list, ignore_index=True).fillna(0)
                areas_trend_df.drop_duplicates(subset=['Year'], keep='last', inplace=True)
                areas_trend_df.sort_values('Year', inplace=True)
                value_vars = [col for col in desired_cols if col in areas_trend_df.columns and col != 'Year']
                areas_long_df = areas_trend_df.melt(id_vars='Year', value_vars=value_vars, var_name='Class', value_name='Area (m²)')
                area_fig = px.bar(areas_long_df, x="Year", y="Area (m²)", color='Class', title="Class Area Composition Over Time",
                                   color_discrete_map={"Water": "#0000FF", "Bareland": "#FFA500", "Mangrove": "#008000", "Prosopis": "#964B00"})

        gain_loss_fig = {"data": [], "layout": {"title": f"No Gain/Loss Data for {year}"}}
        gain_loss_file = os.path.join(CHANGES_DIR, f"Change_Detection_{year-1}_to_{year}_NDVI.csv")
        if os.path.exists(gain_loss_file):
            gain_loss_df = pd.read_csv(gain_loss_file)
            gain_loss_fig = px.bar(gain_loss_df, x="NDVI_Threshold", y=["Gain_m2", "Loss_m2"], title=f"NDVI Change {year-1} to {year}", barmode='group')

        return map_html, preview_src, area_fig, gain_loss_fig

    except Exception as e:
        print(f"--- AN ERROR OCCURRED IN THE RESULTS PAGE CALLBACK ---")
        print(f"INPUTS: Year={year}, Layer='{layer}'")
        traceback.print_exc()
        return error_return

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)