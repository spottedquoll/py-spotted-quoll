from collections import namedtuple
from starburst.routines import get_stored_value


def prepare_inputs(input_data):

    # Prepare inputs
    item_tuple = namedtuple("Item", ['index', 'value', 'weight'])

    lines = input_data.split('\n')

    first_line = lines[0].split()
    item_count = int(first_line[0])
    capacity = int(first_line[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(item_tuple(i - 1, int(parts[0]), int(parts[1])))

    return capacity, items


def prepare_outputs(items, results_vector, apply_original_sorting=True):

    value = get_stored_value(results_vector, items)

    if apply_original_sorting:
        items_original_order = [0] * len(items)

        for idx, val in enumerate(results_vector):
            if val == 1:
                items_original_order[items[idx][0]] = 1

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, items_original_order))

    return output_data