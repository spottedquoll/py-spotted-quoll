from pathlib import Path
import shapefile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import pyexcel as pe
import numpy as np
from numpy import genfromtxt
from quoll.spatial import update_bounding_box


class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


# Paths
top_level_path = Path(__file__).parents[1]
lief_path = Path(__file__).parents[2]
asgs_path = str(lief_path) + '/ASGS/SA2/2011/'

# Get SA2 meta information
sa2_meta_store = []
records = pe.iget_records(file_name=asgs_path + 'SA2_2011_AUST.xlsx')
for row in records:
    sa2_meta_store.append(row)

# Read the shape files
filename = asgs_path + 'SA2_2011_AUST.shp'
sf = shapefile.Reader(filename)
count_shapes = len(list(sf.iterShapes()))

if len(list(sf.iterShapes())) != len(sa2_meta_store):
    raise ValueError('SA2 meta version is different to the shape file version')

# Read the trade matrices for scaling
trade_data = genfromtxt('/Users/jacobfry/Desktop/Project/LIEF/combined_exports_beef_domestic_flows_qld.csv', delimiter=',')
region_totals = np.sum(trade_data,axis=0)

# Set up plot
plt.figure()
ax = plt.axes() # add the axes
ax.set_aspect('equal')
count = 0
bounding_box = None

colour_scaling = MplColorHelper('plasma', np.min(region_totals), np.max(region_totals))

# Read and plot all the shape files
for shape in list(sf.iterShapes()):

    polygon_parts = len(shape.parts)
    if polygon_parts == 1:
        polygon = Polygon(shape.points)

        colour_rgb = colour_scaling.get_rgb(region_totals[count])
        scaled_colours = [colour_rgb[0], colour_rgb[1], colour_rgb[2]]

        patch = PolygonPatch(polygon,fc=(colour_rgb[0], colour_rgb[1], colour_rgb[2],0.7), ec="none")
        ax.add_patch(patch)

        bounding_box = update_bounding_box(bounding_box, shape.bbox)

    elif polygon_parts > 1:
        for ip in range(polygon_parts):  # loop over parts, plot separately
            i0 = shape.parts[ip]
            if ip < polygon_parts - 1:
                i1 = shape.parts[ip + 1] - 1
            else:
                i1 = len(shape.points)

            polygon = Polygon(shape.points[i0:i1 + 1])

            colour_rgb = colour_scaling.get_rgb(region_totals[count])
            scaled_colours = [colour_rgb[0], colour_rgb[1], colour_rgb[2]]

            patch = PolygonPatch(polygon,fc=(colour_rgb[0],colour_rgb[1],colour_rgb[2],0.7), ec="none") # edgecolor=scaled_colours
            ax.add_patch(patch)
            box_limits = update_bounding_box(bounding_box, shape.bbox)

    count = count + 1

plt.xlim(110,155)
plt.ylim(-45,-5)
#plt.show()

plt.savefig('sa2_allocation_whole_country_imports.png', dpi=1200)