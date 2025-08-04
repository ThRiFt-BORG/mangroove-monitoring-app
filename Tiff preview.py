import matplotlib.pyplot as plt
import rasterio

with rasterio.open("data/masks/mask_2019.tif") as src:
    img = src.read(1)

plt.imshow(img, cmap='gray')
plt.title("2018 Mask Preview")
plt.colorbar()
plt.show()
