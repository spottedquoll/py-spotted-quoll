def update_bounding_box(existing_box,new_box):

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