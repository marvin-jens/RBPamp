import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

domains = pd.read_table('domains.txt', header=None, names=['rbp','domains'])
linscore = pd.read_table('scores.txt', header=None, names=['rbp','linscore'])
# descent = pd.read_table('descent2.txt', header=None, names=['rbp','rerr','ferr','corr','steps'])
# descent = pd.read_table('descent_bugfix_noopt.txt', header=None, names=['rbp','rerr','ferr','corr','steps'])
descent = pd.read_table(sys.argv[1], header=None, names=['rbp','rerr','ferr','corr','steps'])
topR = pd.read_table('topR.txt', header=None, names=['rbp','top_R'])
linscore.set_index('rbp')
descent.set_index('rbp')
domains.set_index('rbp')
topR.set_index('rbp')
df = descent.merge(linscore, how='left')
df = df.merge(domains, how='left')
df = df.merge(topR, how='left')
# df = linscore.join(descent, on='rbp', rsuffix='_lin')

def linearity(v):
    if v > 1.:
        return '1+'
    elif v > .7:
        return '0.7-1'
    elif v > 0:
        return '0.7-'
    else:
        return "NA"

def topR(v):
    if v > 20.:
        return '20+'
    elif v > 5:
        return '5-20'
    elif v > 2:
        return '2-5'
    elif v > 1:
        return '1-2'
    else:
        return 'NA'


def domain(dom):
    if type(dom) == float:
        return 'NA'
    from collections import defaultdict
    ds = defaultdict(int)
    for d in dom.split(','):
        ds[d] += 1

    tbl = np.array(sorted([(v,k) for k,v in ds.items()]))[::-1]
    return tbl[0][1]

def label_point(row):
    x,y,val = row
    plt.text(x, y-0.02, str(val))

df['motif_linearity'] = df['linscore'].apply(linearity)
df['domain'] = df['domains'].apply(domain)
df['max_R'] = df['top_R'].apply(topR)
df.set_index('rbp')
print df.loc[df['rbp'] == 'RBFOX2']
rbps = ['RBFOX3','RBFOX2','ELAVL4', 'HNRNPA0','GST','SRSF11','SRSF5','SRSF4','SRSF2','GST','PRR3','ESRP1', 'MSI1']
rows = df.loc[df['rbp'].isin(rbps)]
tolabel = rows[['top_R','corr','rbp']]

corr = df['corr']
qrange = np.percentile(corr,[25,75])
corrmean = np.mean(corr)
print "final best R-value correlation quartile range and mean", qrange, corrmean
lmp = sns.lmplot(x='top_R',y='corr',data=df, fit_reg=False, hue='motif_linearity', hue_order=['1+','0.7-1','0.7-','NA'],legend=True,palette='viridis')
# lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', legend=True)
lmp.set(xscale="log")
tolabel.apply(label_point, axis=1)

plt.xlabel('highest 6mer R-value')
plt.ylabel('final 6mer correlation')
plt.savefig('overview_linearity.pdf')

plt.close()
lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', hue_order=['RRM','KH','ZNF','other','NA'],legend=True)
tolabel.apply(label_point, axis=1)
plt.xlabel('final relative 6mer error')
plt.ylabel('final 6mer correlation')
plt.savefig('overview_domains.pdf')

plt.close()


order = ['20+','5-20','2-5','1-2']
lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='max_R', hue_order=order,legend=True, palette='husl')
tolabel.apply(label_point, axis=1)

plt.xlabel('final relative 6mer error')
plt.ylabel('final 6mer correlation')
plt.savefig('overview_max_R.pdf')
plt.close()

ax = sns.boxplot(x='max_R', y="corr", data=df, order=order[::-1], whis=np.inf, width=.5)
ax = sns.swarmplot(x='max_R', y="corr", data=df, order=order[::-1], color=".2")
plt.xlabel('max. observed 6-mer R-value')
plt.ylabel('max. R-value correlation after fit')
plt.ylim(0,1)
plt.savefig('overview.pdf')


