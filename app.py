import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Config ===
MASK_DIR = "data/masks"
CHANGE_DIR = "data/changes"
OUTPUT_CSV = os.path.join(CHANGE_DIR, "mangrove_area_change.csv")
os.makedirs(CHANGE_DIR, exist_ok=True)

# === 2. Helper to compute area from mask with resolution check (Converting from EPSG:4326{Degrees to Metres})===
def compute_area_from_mask(tif_path):
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

        try:
            pixel_width = transform[0]
            pixel_height = -transform[4]

            if crs.to_string() == "EPSG:4326":   # Reading coordinate system and converting to a metre-based system using 111320 
                if pixel_width == 0 or pixel_height == 0:
                    pixel_area = 100
                else:
                    pixel_width *= 111320
                    pixel_height *= 111320
                    pixel_area = pixel_width * pixel_height
            else:
                pixel_area = pixel_width * pixel_height
        except Exception as e:
            print(f"Using fallback pixel area for {tif_path}: {e}")
            pixel_area = 100

        pixel_count = np.sum(mask == 1)
        area_m2 = pixel_count * pixel_area
        return area_m2

# === 3. Load all mask files ===
years = []
areas = []

for fname in sorted(os.listdir(MASK_DIR)):
    if fname.endswith(".tif") and "mask" in fname.lower():
        year_str = ''.join(filter(str.isdigit, fname))
        if len(year_str) >= 4:
            year = int(year_str[:4])
            path = os.path.join(MASK_DIR, fname)
            area = compute_area_from_mask(path)
            print(f"{fname} → Area (m²): {area}")
            years.append(year)
            areas.append(area)

# === 4. Create DataFrame and compute change ===
df = pd.DataFrame({
    "Year": years,
    "Area_m2": areas
})
df["Area_ha"] = df["Area_m2"] / 10000
df["Area_km2"] = df["Area_m2"] / 1e6
df["Yearly_Change_km2"] = df["Area_km2"].diff()

# === 5. Print and Save ===
print("\nFinal Mangrove Area Summary:")
print(df)
df.to_csv(OUTPUT_CSV, index=False)

# === 6. Plot ===
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Area_m2"], marker='o', label="Mangrove Area (m²)")
plt.bar(df["Year"], df["Area_m2"].diff(), alpha=0.4, label="Change (Δ m²)")
plt.title("Mangrove Area & Yearly Change (2018–2025)")
plt.xlabel("Year")
plt.ylabel("Area (m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CHANGE_DIR, "mangrove_area_trend.png"))
plt.show()
