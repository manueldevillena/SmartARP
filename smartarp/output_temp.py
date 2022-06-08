import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(index=[0])
df['Demand Provision'] = 119156.84
df['Demand Provision Flexible'] = 116059.62
df['Demand Realisation'] = 114875.62

df2 = pd.DataFrame(index=[0])
df2['idle=0, forecast'] = 2438
df2['idle=0, real'] = 2395
df2['idle=60, forecast'] = 20720
df2['idle=60, real'] = 20614

fig, ax = plt.subplots(figsize=(15, 10))
df.plot(ax=ax, kind='bar')

plt.ylabel('Accepted bids')
plt.xticks([])
# plt.axis([-0.5,0.5,112000,125000])
plt.tight_layout()
plt.legend(fontsize=15)
plt.savefig('/home/villena/bitbucket/iaee_presentation/images/test2.pdf')
plt.close(fig)



