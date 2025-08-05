import os
from matplotlib import pyplot as plt
import numpy as np
import rasterio
import pandas as pd

#Indices directory
NDVI_DIR = "data/ndvi"
NDMI_DIR = "data/ndmi"
OUTPUT_DIR = "data/changes"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ndvi_ndmi_stats.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "ndvi_ndmi_stats.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)


#Mean stats calculation
def calculate_mean_from_tif(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        data[data< -1]= np.nan
        data[data> 1] = np.nan
        return float(np.nanmean(data));

# Loop through the years
years = []
ndvi_means = []
ndmi_means = []

for year in range (2018, 2025):
    ystr = str(year)
    ndvi_path = os.path.join(NDVI_DIR, f"ndvi_{ystr}.tif") # Path and format of naming for data
    ndmi_path = os.path.join(NDMI_DIR, f"ndmi_{ystr}.tif");

    if os.path.exists(ndvi_path) and os.path.exists(ndmi_path):
        ndvi_mean = calculate_mean_from_tif(ndvi_path)
        ndmi_mean = calculate_mean_from_tif(ndmi_path)
        print(f"{year} → NDVI: {ndvi_mean:.3f}, NDMI: {ndmi_mean:.3f}")
        years.append(year)
        ndvi_means.append(ndvi_mean)
        ndmi_means.append(ndmi_mean)
    else:
        print(f"Skipping {year} - One or both files missing");

# Create DataFrame
df = pd.DataFrame({
    "Year" : years,
    "Ndvi_Mean" : ndvi_means,
    "Ndmi_Mean" : ndmi_means
})

print("/n NDVI & NDMI Summary:" )
print(df)

# Save CSV 
df.to_csv(OUTPUT_CSV, index = False)

# Plot Results
plt.figure(figsize = (10,5))
plt.plot(df["Year"], df["Ndvi_Mean"], marker='o', label="NDVI Mean", color='green')
plt.plot(df["Year"], df["Ndmi_Mean"], marker='s', label="NDMI Mean", color='blue')
plt.title("NDVI & NDMI Mean per year between (2018-2025)")
plt.xlabel("Year")
plt.ylabel("Mean Index Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
plt.show


