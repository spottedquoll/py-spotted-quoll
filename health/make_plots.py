import pandas as pd
import matplotlib.pyplot as plt

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

p2 = plt.barh(x_pos, b, label=series_labels[1], color='xkcd:salmon')
p1 = plt.barh(x_pos, a, label=series_labels[0], color='xkcd:dark sky blue')

plt.xlabel('Total GHG emissions for medical service expenditure (kt CO$_2$-e)', labelpad=10)
plt.yticks(x_pos, labels)
plt.legend(loc='upper right', frameon=False)
plt.grid(which='major', axis='x', linestyle='--')

plt.savefig(work_dir + 'figs/fig2_bar_chart.png', dpi=700, bbox_inches='tight')
plt.savefig(work_dir + 'figs/fig2_bar_chart.pdf', bbox_inches='tight')

plt.clf()