
import pandas as pd

pd_all = pd.read_csv('waimai_10k.csv')

print('總共評論數目：%d' % pd_all.shape[0])
print('正向：%d' % pd_all[pd_all.label==1].shape[0])
print('負向：%d' % pd_all[pd_all.label==0].shape[0])
print(pd_all.sample(20))
