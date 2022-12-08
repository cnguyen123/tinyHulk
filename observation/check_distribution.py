import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('../data/csv_data/meccano_bike/back.csv')
areas = []
diognosis = []
height = []
for i in range(0, len(df)):
    xmin = df.loc[i, 'xmin']
    ymin = df.loc[i, 'ymin']
    xmax = df.loc[i, 'xmax']
    ymax = df.loc[i, 'ymax']
    areas.append( (xmax -xmin)*(ymax-ymin))
    diognosis.append(math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2))
    height.append((xmax-xmin))


#plt.plot(areas, color='r')
#plt.title('area')
#plt.tight_layout()
#plt.show()

#plt.plot(diognosis, color='r')
#plt.title("'diognosis'")
#plt.tight_layout()
#plt.show()

plt.plot(height, color='r')
plt.title("'height'")
plt.tight_layout()
plt.show()
s = pd.Series(areas)
print(s.describe())