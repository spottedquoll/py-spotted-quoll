import pandas as pd
import matplotlib.pyplot as plt

work_dir = '/Users/jacobfry/Desktop/japan_health/'
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