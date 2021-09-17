import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/dimension_data.csv')
sns.regplot(x="n_dim", y="n_correct", data=data)
plt.show()