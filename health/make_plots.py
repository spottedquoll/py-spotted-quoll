import pandas as pd
import matplotlib.pyplot as plt
from quoll.draw import discrete_cmap

print('making plots...')

work_dir = '/Volumes/slim/Projects/uni_projects/japan_health/'
file = work_dir + 'FigureData.xlsx'
xl = pd.ExcelFile(file)

# figure 1
df1 = xl.parse('Fig1Data')
data = df1['CarbonF'].values
labels = df1['Category'].values

plt.figure()
plt.pie(data, labels=labels)
plt.savefig(work_dir + 'figs/fig1_pie.png', dpi=700)
plt.clf()

# figure 2
df1 = xl.parse('Fig2Data')
df_tr = df1.transpose()

labels = df1['Disease'].values
series_labels = list(df1)
del series_labels[0]

a = df1[series_labels[0]].values
b = df1[series_labels[1]].values

b = b + a

a = a/1000
b = b/1000

x_pos = [i for i, _ in enumerate(labels)]  # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

plt.figure()
ax = plt.subplot(111)

p2 = plt.barh(x_pos, b, label=series_labels[1], color='xkcd:salmon', edgecolor='xkcd:slate grey', linewidth=0.5)
p1 = plt.barh(x_pos, a, label=series_labels[0], color='xkcd:dark sky blue', edgecolor='xkcd:slate grey', linewidth=0.5)

plt.xlabel('Total GHG emissions for medical service expenditure (kt CO$_2$-e)', labelpad=10)
plt.yticks(x_pos, labels)
plt.grid(which='major', axis='x', linestyle='--')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, ncol=2)

plt.savefig(work_dir + 'figs/fig2_bar_chart.png', dpi=700, bbox_inches='tight')
plt.savefig(work_dir + 'figs/fig2_bar_chart.pdf', bbox_inches='tight')

plt.clf()

# Figure 3
df1 = xl.parse('Fig3Data')

labels = df1['Disease'].values
series_labels = list(df1)
del series_labels[0]

a = df1[series_labels[0]].values
b = df1[series_labels[1]].values

a = a/1000
b = b/1000

x_pos = [i for i, _ in enumerate(labels)]  # the x locations for the groups
bar_width = 0.35       # the width of the bars: can also be len(x) sequence
x_pos_2 = [x + bar_width for x in x_pos]
tick_positions = [r + 0.5*bar_width for r in range(len(labels))]

plt.figure()
ax = plt.subplot(111)

p2 = plt.barh(x_pos, a, height=bar_width, label=series_labels[1], color='xkcd:cool green'
              , edgecolor='xkcd:slate grey', linewidth=0.3)
p1 = plt.barh(x_pos_2, b, height=bar_width, label=series_labels[0], color='xkcd:lilac'
              , edgecolor='xkcd:slate grey', linewidth=0.3)

plt.xlabel('Total GHG emissions per patient (t CO$_2$-e/pp)', labelpad=10)
plt.yticks(tick_positions, labels)
# plt.legend(loc='upper right', frameon=False)
plt.grid(which='major', axis='x', linestyle='--')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, ncol=2)

plt.savefig(work_dir + 'figs/fig3_bar_chart.png', dpi=700, bbox_inches='tight')
plt.savefig(work_dir + 'figs/fig3_bar_chart.pdf', bbox_inches='tight')

plt.clf()

# Figure 4
df1 = xl.parse('Fig4DataFlipped')
scale = 10e6

x_labels = df1['Year'].values
series_labels = list(df1)
last_category = series_labels[-1]
del series_labels[-1], series_labels[0]

x_pos = [i for i, _ in enumerate(x_labels)]  # the x locations for the groups

total_array = None
scatter_data = []
width = 0.4

clrs = plt.cm.get_cmap('tab20')

plt.figure()
ax = plt.subplot(111)
last = None

# Make stacked bar chart
for idx, series in enumerate(series_labels):

    # select series
    data = df1[series].values
    data = data/scale

    plt.bar(x_pos, data, width, edgecolor='xkcd:slate grey', linewidth=0.5, label=series, bottom=last
            , color=plt.cm.Set3(idx))

    if last is None:
        last = data
    else:
        last = last + data

plt.xticks(x_pos, x_labels)
ax.set_ylabel('Induced GHG emissions per patient (MT CO$_2$-e)', labelpad=10)

# Overlay scatter plot
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.scatter(x_pos, df1[last_category].values, label='Share of the national total emissions', alpha=0.6, c='red')
ax2.set_ylabel('Share of national GHG emissions (%)', labelpad=10)
ax2.set_ylim(4.3, 5)

ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), frameon=False, ncol=2)
ax2.legend(loc='lower center', bbox_to_anchor=(0.05, -0.17), frameon=False)

plt.savefig(work_dir + 'figs/fig4_bar_chart.png', dpi=700, bbox_inches='tight')
plt.savefig(work_dir + 'figs/fig4_bar_chart.pdf', bbox_inches='tight')

plt.clf()

print('finished')