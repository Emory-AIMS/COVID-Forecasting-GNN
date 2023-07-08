import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = "ca48-548"
filepath = f'figures/{dataset}-pcc.png'

ts = np.loadtxt(open("../data/ts/{}.txt".format(dataset)), delimiter=',')
pcc = np.corrcoef(ts, rowvar=False)
print(ts.shape)
print(pcc.shape)

plt.figure(0)
sns.heatmap(pcc)
plt.title("Pearson Correlation for the 48 CA Counties")
plt.savefig(filepath)
plt.close(0)

threshold = 0.7
filepath = f'figures/{dataset}-{threshold}-pcc.png'

pcc_filtered = pcc * (pcc > threshold) 
plt.figure(1)
sns.heatmap(pcc_filtered)
plt.title(f'Pearson Correlation for the 48 CA Counties: Threshold = {threshold}')
plt.savefig(filepath)
plt.close(1)