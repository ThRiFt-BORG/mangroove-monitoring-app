import matplotlib.pyplot as plt
import rasterio

with rasterio.open("data/ndvi/ndvi_2019.tif") as src:
    img = src.read(1)

plt.imshow(img, cmap='Greens')
plt.title("2019 NDVI Preview")
plt.colorbar()
plt.show()
