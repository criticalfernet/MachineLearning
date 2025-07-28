import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotstyle import plotstyle

dimMap = {
    1:"OverWorld",
    2:"Nether",
    3:"End"
}

df = pd.read_csv('Biomes.csv')
df = df.drop(columns=['minecraftVersion'],errors='ignore')
df['precipitation'] = df['precipitation'].fillna('None') 
df['dimensionID'] = df['dimensionID'].map(dimMap)

data_list = df.to_dict('records')

idlist = [x['ID'] for x in data_list]
trees = [x['treesOrGrass'] for x in data_list]

x = np.array(idlist)
y = np.array(trees)

plt.rcParams.update(plotstyle)

_, ax = plt.subplots()

ax.scatter(x, y, color='#0f67a4',marker='o', s=75, edgecolor='#dadada')
ax.set_xlabel('Biome Id')
ax.set_ylabel('Contains Trees')
ax.grid(True, linestyle='--', alpha=0.8)

plt.title('Biomes and Trees', fontsize=14, pad=15)

plt.show()

