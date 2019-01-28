from collections import namedtuple


def greedy(items, capacity):

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full

    value = 0
    weight = 0
    taken = [0] * len(items)

    for index, item in enumerate(items):
        if weight + item.weight <= capacity:
            taken[index] = 1
            value += item.value
            weight += item.weight

    return taken


def validate_results(results_vector, items, capacity):

    if len(results_vector) != len(items):
        raise ValueError('Results dimension mismatch')

    print('Solution value: ' + str(get_stored_value(results_vector, items)))

    acc = 0

    for idx, val in enumerate(results_vector):
        if val == 1:
            acc = acc + items[idx][2]

    if acc <= capacity:
        print('Results are feasible, weight: ' + str(acc))
        return True
    else:
        print('Results are infeasible!! weight: ' + str(acc))
        return False


def score_results(results_vector, items):

    acc = 0

    for idx, val in enumerate(results_vector):
        if val == 1:
            acc = acc + items[idx][1]

    print('Score: ' + str(acc))
    return acc


def map_item_density(items):

    items_inc_density = []
    seen = []
    items_density = namedtuple("Item", ['index', 'value', 'weight', 'density'])
    for i in items:
        density = i[1] / i[2]
        items_inc_density.append(items_density(i[0], int(i[1]), int(i[2]), density))

        if density not in seen:
            seen.append(density)
        else:
            print('Duplicate density: ' + str(density))

    return items_inc_density


def branch_and_bound(items, capacity):

    feasible_solutions = []

    # Guess the solution
    guess_vector = greedy(items, capacity)
    best_estimate = get_stored_value(guess_vector, items)
    print('Initial guess:' + str(best_estimate))
    validate_results(guess_vector, items, capacity)
    feasible_solutions.append({'path_vector': guess_vector, 'value': best_estimate})

    # Attempt to find a better solution
    feasible_solutions, best_estimate = traverse_branch(items, best_estimate, [], feasible_solutions, capacity, capacity,
                                                        branch_direction='left')

    feasible_solutions, best_estimate = traverse_branch(items, best_estimate, [], feasible_solutions, capacity, capacity,
                                                        branch_direction='right')

    # Select the best result
    best_path_vector = [element for element in feasible_solutions if element['value'] == best_estimate][0]['path_vector']
    value = get_stored_value(best_path_vector, items)

    print('Best solution: ' + str(value))

    return best_path_vector


def traverse_branch(items, best_estimate, path_vector, solutions, remaining_capacity, total_capacity, branch_direction):

    # Define the current choice
    if len(path_vector) == len(items):
        return solutions, best_estimate
    elif branch_direction == 'left':
        new_path_vector = path_vector + [1]
    else:
        new_path_vector = path_vector + [0]

    # Check feasibility
    if get_current_weight(items, new_path_vector) > remaining_capacity:
        return solutions, best_estimate

    # Compute the bound
    max_branch_value = estimate_max_branch_value(items, new_path_vector, total_capacity)

    # Check bound:
    if best_estimate >= max_branch_value:
        return solutions, best_estimate

    # If this a leaf, return the current solution
    if len(new_path_vector) == len(items):
        total_value = get_stored_value(new_path_vector, items)
        solutions.append({'path_vector': new_path_vector, 'value': total_value})
        print('New best guess:' + str(total_value))
        return solutions, total_value

    else:
        # Otherwise continue branching
        solutions, best_estimate = traverse_branch(items, best_estimate, new_path_vector, solutions, remaining_capacity, total_capacity,
                                                   branch_direction='left')

        solutions, best_estimate = traverse_branch(items, best_estimate, new_path_vector, solutions, remaining_capacity, total_capacity,
                                                   branch_direction='right')

        return solutions, best_estimate


def estimate_max_branch_value(items, results_vector, total_capacity):

    """
        iterates from depth + 1 to n, adding to the value and subtracting from the capacity
        until there is no more capacity and fill in the remaining fractional part.
    """

    current_value = get_stored_value(results_vector, items)

    # Have we reached the end of the branch already?
    if len(results_vector) == len(items):
        return current_value
    else:

        path_vector_temp = results_vector

        # Simulate the maximum value path vector
        count_of_items = len(path_vector_temp)
        cumulative_weight = get_current_weight(items, path_vector_temp)

        for index, item in enumerate(items):
            if index > count_of_items-1:
                if cumulative_weight + item.weight <= total_capacity:
                    current_value += item.value
                    cumulative_weight += item.weight
                elif cumulative_weight < total_capacity:
                    remaining_capacity = total_capacity - cumulative_weight
                    fraction = remaining_capacity / item.weight
                    current_value += (item.value * fraction)
                    cumulative_weight += (item.weight*fraction)

        if cumulative_weight > total_capacity:
            raise ValueError('Allowed weight exceeded!')

        return current_value


def get_current_weight(items, results_vector):

    acc = 0

    for idx, val in enumerate(results_vector):
        if val == 1:
            acc = acc + items[idx][2]

    return acc


def get_stored_value(results_vector, items):

    acc = 0

    for idx, val in enumerate(results_vector):
        if val == 1:
            acc = acc + items[idx][1]

    return acc


def get_results_vector(result_str):

    result_spl = result_str.split('\n')[1]

    return [int(s) for s in result_spl.split(' ')]

