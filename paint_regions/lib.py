import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import numpy as np


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def colour_polygons_by_vector(colour_scale_data, all_shapes, sub_regions, save_file_name, bounding_box
                              , normalise=True, colour_map='plasma'):

    if normalise:
        colour_scale_data = colour_scale_data / sum(colour_scale_data)
        if sum(colour_scale_data) > 1.001 or sum(colour_scale_data) < 0.999:
            raise ValueError('Normalisation failed')
        min_val = 0
        max_val = 1
    else:
        min_val = np.min(colour_scale_data)
        max_val = np.max(colour_scale_data)

    colour_scaling = MplColorHelper(colour_map, min_val, max_val)

    if len(colour_scale_data) != len(sub_regions):
        raise ValueError('SA2 count does not match trade data dimension')

    count = 0

    # Set up plot
    plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    # Read and plot all the shape files
    for index in sub_regions:

        shape = all_shapes[int(index) - 1]
        polygon_parts = len(shape.parts)

        colour_rgb = colour_scaling.get_rgb(colour_scale_data[count])
        # scaled_colours = [colour_rgb[0], colour_rgb[1], colour_rgb[2]]

        if polygon_parts == 1:
            polygon = Polygon(shape.points)

            try:
                patch = PolygonPatch(polygon, fc=(colour_rgb[0], colour_rgb[1], colour_rgb[2], 0.7), ec="none")
                ax.add_patch(patch)
            except:
                stop = 1

        elif polygon_parts > 1:
            for ip in range(polygon_parts):  # loop over parts, plot separately
                i0 = shape.parts[ip]
                if ip < polygon_parts - 1:
                    i1 = shape.parts[ip + 1] - 1
                else:
                    i1 = len(shape.points)

                polygon = Polygon(shape.points[i0:i1 + 1])

                patch = PolygonPatch(polygon, fc=(colour_rgb[0], colour_rgb[1], colour_rgb[2], 0.7), ec="none")
                ax.add_patch(patch)

        count = count + 1

    plt.xlim(bounding_box[0], bounding_box[1])
    plt.ylim(bounding_box[2], bounding_box[3])

    plt.savefig(save_file_name, dpi=1400)
    plt.clf()
    plt.close("all")

    print('.')