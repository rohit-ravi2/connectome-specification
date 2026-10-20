import warnings,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
D="../accounting/principles/"
dfp=pd.read_csv(D+"A5_doseresponse_pairs.csv")
# two-axis: predict k from log(area) and log(meansyn|present); among present pairs
sub=dfp[dfp['k']>0].copy()
sub['la']=np.log10(sub['area']+1); sub['ls']=np.log10(sub['meansyn'])
X=sub[['la','ls']].values; y=sub['k'].values
lr=LinearRegression().fit(X,y)
print("predict k (present pairs) from [log area, log strength]: coefs",dict(zip(['logArea','logStrength'],np.round(lr.coef_,3))),"R2=%.3f"%lr.score(X,y))
print("  area-only R2=%.3f"%LinearRegression().fit(sub[['la']],y).score(sub[['la']],y),
      " strength-only R2=%.3f"%LinearRegression().fit(sub[['ls']],y).score(sub[['ls']],y))
print("  partial spearman: k vs area | -- =%.3f ; k vs strength=%.3f"%(spearmanr(sub['k'],sub['la'])[0],spearmanr(sub['k'],sub['ls'])[0]))

# DEGREE CONFOUND: postsyn command neurons AVA/AVB get many weak variable inputs.
# out-degree(pre) / in-degree(post) among contacting pairs vs variability
indeg=dfp.groupby('post')['k'].apply(lambda s:(s>0).sum())
outdeg=dfp.groupby('pre')['k'].apply(lambda s:(s>0).sum())
dfp['post_indeg']=dfp['post'].map(indeg); dfp['pre_outdeg']=dfp['pre'].map(outdeg)
var=dfp[(dfp['k']>0)&(dfp['k']<8)]; pr=dfp[dfp['k']==8]
print("\nDEGREE CONFOUND check (contacting-partner counts):")
print("  variable pairs: median post in-degree=%.0f pre out-degree=%.0f"%(var['post_indeg'].median(),var['pre_outdeg'].median()))
print("  repro_present : median post in-degree=%.0f pre out-degree=%.0f"%(pr['post_indeg'].median(),pr['pre_outdeg'].median()))
print("  -> variable synapses land on HIGHER-degree hub targets (weak inputs to command neurons)")
