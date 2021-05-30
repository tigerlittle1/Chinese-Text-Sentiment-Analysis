
import pandas as pd
from tqdm import tqdm
from opencc import OpenCC


# pd_all = pd.read_csv('ChnSentiCorp_htl_all.csv')
pd_all=pd.read_csv('ChnSentiCorp_htl_all.csv', names=['label', 'review'])

# print('總共評論數目：%d' % pd_all.shape[0])
# print('正向：%d' % pd_all[pd_all.label==1].shape[0])
# print('負向：%d' % pd_all[pd_all.label==0].shape[0])
# # print(pd_all.sample(20))
# print(pd_all["label"][0],pd_all["review"][0])
# cc = OpenCC("s2twp")
# print(cc.convert(pd_all["review"][0]))

max = 0
for index , (y,x) in tqdm(pd_all.iterrows()):

    try:
        if len(x)>max:
            max = len(x)
    except:
        print(index, y,str(x))
    #print(x,y)

print(max)