import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy import array, concatenate

# Directories
data_dir = os.environ['streak_dir']
save_dir = data_dir + '/plots/'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Read Habithub data
habits = pd.read_csv(data_dir + 'Habithub.csv')
habits.head()

# Build streaks
subset = habits[habits.Name == 'P']

i = 0  # streak_counter
j = 0  # streak_number
k = 0  # number in series
results = []

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

        results.append([i, j, k])

streaks_df = pd.DataFrame(results, columns=['streak_count', 'streak_num', 'member_num'])
streaks_df.to_csv(data_dir + '/processed_streaks.csv', index=False)

# Plot
fig, ax = plt.subplots()  # Create figure object

for s in streaks_df['streak_num'].unique():
    subset = streaks_df[streaks_df.streak_num == s]
    if len(subset) > 1:

        # Line segments
        points = array([x_sample, y_sample]).T.reshape(-1, 1, 2)
        segments = concatenate([points[:-1], points[1:]], axis=1)
        ax.add_collection(LineCollection(segments, linewidths=line_thickness_sample
                                         , color='xkcd:' + the_colours[idx], alpha=line_seg_alpha
                                         , linestyle=ls))

# Save
plot_fname = ('malaria_forestry_trends_c' + str(i) + '_s' + str(s) + '_pl' + str(pl) + '_ft'
              + str(ft) + '.png')
fig.savefig(save_dir + plot_fname, dpi=700, bbox_inches='tight')

# Finished with this figure
plt.clf()
plt.close(fig)

# For meteors, could just use lines. It seems difficult to achieve both changing thickness and opacity

# Line segments are good for random walks
