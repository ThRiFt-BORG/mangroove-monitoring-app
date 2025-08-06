# Mangrove Monitoring Dashboard

A Python-based dashboard for monitoring mangrove forest changes using satellite imagery analysis. This project analyzes NDVI (Normalized Difference Vegetation Index), NDMI (Normalized Difference Moisture Index), and calculates mangrove area changes over time.

## Features

- Interactive dashboard showing mangrove area trends
- NDVI and NDMI analysis for vegetation health monitoring
- Time series visualization (2018-2025)
- Raster image preview for visual analysis
- CSV export of calculated statistics

## Prerequisites

Before setting up this project, ensure you have the following installed on your system:

- **Python 3.7 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (Python package installer) - Usually comes with Python
- **Git** (optional) - For cloning the repository

## Project Structure

```
mangrove-monitoring/
├── data/
│   ├── masks/          # Mangrove mask TIF files
│   ├── ndvi/           # NDVI TIF files  
│   ├── ndmi/           # NDMI TIF files
│   └── changes/        # Generated CSV files
├── exports/            # Generated PNG files
├── app.py              # Main dashboard application
├── Calculations.py     # Area calculation script
├── Indices_Summary.py  # NDVI/NDMI analysis script
├── Tiff preview.py     # Quick TIF preview script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation & Setup

### Step 1: Clone or Download the Project

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/ThRiFt-BORG/mangroove-monitoring-app
cd mangroove-monitoring-app
```

**Option B: Download ZIP**
1. Download the project as a ZIP file
2. Extract it to your desired location
3. Open terminal/command prompt in the extracted folder

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment helps avoid conflicts with other Python projects:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt when the virtual environment is active.

### Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

If you encounter any issues, try upgrading pip first:
```bash
pip install --upgrade pip
```

### Step 4: Prepare Your Data

Create the required data directory structure and add your TIF files:

```bash
mkdir -p data/masks data/ndvi data/ndmi data/changes exports
```

**File Naming Convention:**
- NDVI files: `ndvi_YYYY.tif` (e.g., `ndvi_2019.tif`)
- NDMI files: `ndmi_YYYY.tif` (e.g., `ndmi_2019.tif`) 
- Mask files: `mask_YYYY.tif` (e.g., `mask_2019.tif`)

Place your TIF files in the corresponding directories:
- `data/ndvi/` - for NDVI raster files
- `data/ndmi/` - for NDMI raster files  
- `data/masks/` - for mangrove mask files

## Usage

### Running Individual Analysis Scripts

**1. Calculate Mangrove Areas:**
```bash
python Calculations.py
```
This script will:
- Process all mask files in `data/masks/`
- Calculate areas in m², hectares, and km²
- Generate `mangrove_area_change.csv` 
- Create a trend visualization

**2. Analyze NDVI/NDMI Statistics:**
```bash
python Indices_Summary.py
```
This script will:
- Calculate mean NDVI and NDMI values for each year
- Generate `ndvi_ndmi_stats.csv`
- Create comparison charts

**3. Preview Individual TIF Files:**
```bash
python "Tiff preview.py"
```
Quick preview of a single NDVI file (modify the script to change the file).

### Running the Interactive Dashboard

Launch the main dashboard application:

```bash
python app.py
```

The dashboard will start running locally. Open your web browser and navigate to:
```
http://127.0.0.1:8050/
```

## Dashboard Features

Once the dashboard is running, you can:

- **Select Year**: Use the dropdown to view data for different years
- **View Trends**: Interactive graphs showing NDVI/NDMI trends and area changes
- **Raster Preview**: Visual comparison of NDVI, NDMI, and mask data
- **Export Data**: Generated CSV files are saved in `data/changes/`

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install --upgrade -r requirements.txt
```

**2. GDAL/Rasterio Installation Issues**
On Windows, if you encounter GDAL-related errors:
```bash
pip install --find-links https://girder.github.io/large_image_wheels GDAL rasterio
```

**3. Permission Errors**
Make sure you have write permissions in the project directory, especially for the `data/changes/` and `exports/` folders.

**4. Port Already in Use**
If port 8050 is busy, modify `app.py` and change the port:
```python
app.run(debug=True, port=8051)
```

**5. Missing Data Files**
Ensure your TIF files follow the exact naming convention and are placed in the correct directories.

### Performance Tips

- **Large Files**: If working with large TIF files, consider using a machine with sufficient RAM (8GB+ recommended)
- **File Formats**: Ensure TIF files are properly formatted GeoTIFF files
- **Coordinate Systems**: The code handles EPSG:4326 coordinate systems with automatic conversion

## Data Requirements

### Input Data Specifications

- **File Format**: GeoTIFF (.tif)
- **Coordinate System**: EPSG:4326 (WGS84) preferred
- **Data Type**: Float32 or similar for NDVI/NDMI (-1 to 1 range)
- **Mask Values**: Binary (0 = no mangrove, 1 = mangrove)

### Expected Years

The current setup expects data from 2018-2025, but you can modify the year ranges in:
- `Indices_Summary.py` (line 25): `for year in range(2018, 2026)`
- Add/remove files as needed

## Customization

### Modifying Year Range
Edit the range in `Indices_Summary.py`:
```python
for year in range(2020, 2024):  # Change as needed
```

### Changing Color Schemes
Modify the colormaps in `app.py`:
```python
cmap = {
    "ndvi": "Greens",    # Change to "RdYlGn", "viridis", etc.
    "ndmi": "Blues",     # Change to "plasma", "coolwarm", etc.
    "mask": "gray"
}.get(label, "viridis")
```

### Adding New Indices
Follow the pattern in `Indices_Summary.py` to add additional vegetation indices.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify your data files are in the correct format and location
4. Check Python and package versions compatibility

Author:Kimani
