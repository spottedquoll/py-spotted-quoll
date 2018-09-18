import shapefile
import pyexcel as pe


def update_bounding_box(existing_box, new_box):
    if existing_box is None:
        return new_box

    # xmin
    if new_box[0] < existing_box[0]:
        existing_box[0] = new_box[0]

    # ymin
    if new_box[1] < existing_box[1]:
        existing_box[1] = new_box[1]

    # xmax
    if new_box[2] > existing_box[2]:
        existing_box[2] = new_box[2]

    # ymax
    if new_box[3] > existing_box[3]:
        existing_box[3] = new_box[3]

    return existing_box


def read_shape_files(filename):
    # Read the shape files
    sf = shapefile.Reader(filename)
    all_shapes = list(sf.iterShapes())

    return all_shapes


def read_sa2_meta(filename):
    sa2_meta_store = []
    records = pe.iget_records(file_name=filename)
    for row in records:
        sa2_meta_store.append(row)

    return sa2_meta_store