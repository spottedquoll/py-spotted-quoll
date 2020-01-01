import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy import array, concatenate, random, linspace
from streak_plotter.lib import symmetric_log
from quoll.discrete_colour_lists import kyoto_colours

# Directories
data_dir = os.environ['streak_dir']
save_dir = data_dir + '/plots/'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

print('Reading streak data')

# Read Habithub data
habits = pd.read_csv(data_dir + 'Habithub.csv')
habits.head()

# Get habits
select_habits_str = os.environ['select_habits']
select_habits = select_habits_str.split(',')

assert len(select_habits) > 0

# Build streaks
print('Extracting streaks')
results = []

for h in select_habits:

    subset = habits[habits.Name == h]

    i = 0  # streak_counter
    j = 0  # streak_number
    k = 0  # number in series

    for index, row in subset.iterrows():
        if type(row['Description']) == str and not row['Current Streak'] == 'SKIP':
            k = k + 1

            if row['Current Streak'] == 'DONE':
                i = i + 1
            elif row['Current Streak'] == 'FAIL':
                j = j + 1
                i = 0
            else:
                raise ValueError('Unknown data value')

            results.append([h, i, j, k])

streaks_df = pd.DataFrame(results, columns=['habit', 'streak_count', 'streak_num', 'member_num'])
streaks_df.to_csv(data_dir + '/processed_streaks.csv', index=False)

# Meteors
print('Plotting meteors')

# Options
image_quality = 800  # dpi
y_is_walk = False
y_variation_step = 0.03
the_colours = kyoto_colours()
z = 0

# Create figure
fig, ax = plt.subplots()  # Create figure object
x_limits = [0, 1]
y_limits = [0, 0]

for h in select_habits:

    print('Plotting ' + h)

    previous_y_offset = random.uniform(-1*y_variation_step, y_variation_step)
    steaks_h = streaks_df[streaks_df.habit == h]

    print('Found ' + str(len(steaks_h['streak_num'].unique())) + ' streaks')

    for s in steaks_h['streak_num'].unique():
        subset = steaks_h[steaks_h.streak_num == s]
        if len(subset) > 1:

            x = subset['member_num'].values + random.uniform(-4, 0)
            b = previous_y_offset + random.uniform(-1 * y_variation_step, y_variation_step)

            if y_is_walk:
                b = b + previous_y_offset
                previous_y_offset = b

            y = linspace(b, b, len(x))

            streak_length = max(subset['member_num'].values)/100
            line_thickness = symmetric_log(subset['streak_count'])

            # Line segments
            points = array([x, y]).T.reshape(-1, 1, 2)
            segments = concatenate([points[:-1], points[1:]], axis=1)
            ax.add_collection(LineCollection(segments, linewidths=line_thickness, color=the_colours[z], alpha=0.8))

            x_limits = [min(x[0], x_limits[0]), max(x[1], x_limits[1])]
            y_limits = [min(y[0], y_limits[0]), max(y[1], y_limits[1])]

    z = z + 1

# Limits
plt.ylim(bottom=0.95 * y_limits[0], top=1.05 * y_limits[1])
plt.xlim(left=0.95 * x_limits[0], right=1.05 * x_limits[1])

# Frame
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

# Save
plot_fname = (save_dir + 'meteor' + '.png')
fig.savefig(plot_fname, dpi=image_quality, bbox_inches='tight')

# Finished with this figure
plt.clf()
plt.close(fig)

print('Finished')