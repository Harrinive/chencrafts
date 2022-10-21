from matplotlib import pyplot as plt
from matplotlib import colormaps

def filter(c, filter_name):
    if filter_name in ["translucent", "trans"]:
        r, g, b, a = c
        return [r, g, b, a * 0.2]
    elif filter_name in ["emphsize", "emph"]:
        r, g, b, a = c
        factor = 3
        return [r ** factor, g ** factor, b ** factor, a]

class IntCmap():
    def __init__(self, total, cmap_name="rainbow"):
        self.total = total
        self.cmap_name = cmap_name

        self.cmap = colormaps[self.cmap_name]
        self.norm = plt.Normalize(0, self.total)
    
    def __call__(self, idx):
        return self.cmap(self.norm(idx))