import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

recipe_directory = '/Volumes/slim/2017_ProductionRecipes/'

# Import data set
xl = pd.ExcelFile(recipe_directory + 'results/national_IOT_sector_counts.xlsx')
df1 = xl.parse(xl.sheet_names[0], parse_dates=False)

# Clean
df1.fillna(0, inplace=True)

# column labels
year_columns = list(df1)
remove_columns = ['Acronym', 'Root number', 2017, 2016, 2013]
for c in remove_columns:
    year_columns.remove(c)
acro_labels = list(df1['Acronym'])

# vectors defining histogram
y_len = len(acro_labels)
x_len = len(year_columns)

x = np.zeros(x_len * y_len, dtype=int)
y = np.zeros(x_len * y_len, dtype=int)
c = np.zeros(x_len * y_len, dtype=float)
table = np.zeros((x_len, y_len), dtype=float)
count = 0

for row_index, row in df1.iterrows():
    acronym_index = acro_labels.index(row['Acronym'])
    for col_index, sectors in row.iteritems():
        if col_index in year_columns:
            year_index = year_columns.index(col_index)
            if not isinstance(sectors, float) and not isinstance(sectors, int):
                if isinstance(sectors, datetime.datetime):
                    sectors = sectors.strftime('%m/%y')
                if '/' not in sectors:
                    raise ValueError('Cannot parse sector count')
                else:
                    elements = sectors.split('/')
                    multiple_counts = []
                    for e in elements:
                        multiple_counts.append(float(e))
                    sectors = max(multiple_counts)

            x[count] = year_index
            y[count] = acronym_index
            if sectors != 0:
                c[count] = sectors
            table[year_index, acronym_index] = sectors
            count = count + 1

# Make the plot
# plt.hexbin(x, y, c, gridsize=25)
# plt.xticks(np.arange(x_len), year_columns, rotation=70)
#plt.yticks(np.arange(y_len), acro_labels)
# plt.xlabel('Year')
# plt.ylabel('Country')
# plt.show()

# Replace zeros
table[table <= 0] = 0.001

# make x labels
x_labels =[]
x_tick_marks = []
counter = 0
step = 1
for idx, item in enumerate(year_columns):
    if counter == step:
        x_tick_marks.append(idx)
        x_labels.append(item)
        if counter == step:
            counter = 0
        else:
            counter = counter + 1
    else:
        counter = counter + 1

# make y labels
y_labels =[]
y_tick_marks = []
counter = 0
step = 3
for idx, item in enumerate(acro_labels):
    if counter == step:
        y_tick_marks.append(idx)
        y_labels.append(item)
        if counter == step:
            counter = 0
        else:
            counter = counter + 1
    else:
        counter = counter + 1

# Standard heatmap
plt.figure()
fig, ax = plt.subplots()
plt.imshow(np.transpose(table), aspect='auto', cmap='plasma')
plt.xticks(x_tick_marks, x_labels, rotation=70, fontsize=6)
plt.yticks(y_tick_marks, y_labels, fontsize=6)
plt.colorbar()
plt.savefig(recipe_directory + 'results/sector_count_heatmap.png', dpi=700)
plt.clf()