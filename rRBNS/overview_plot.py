import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

domains = pd.read_table('domains.txt', header=None, names=['rbp','domains'])
linscore = pd.read_table('scores.txt', header=None, names=['rbp','linscore'])
topR = pd.read_table('topR.txt', header=None, names=['rbp','top_R'])
linscore.set_index('rbp')
domains.set_index('rbp')
topR.set_index('rbp')

# df = linscore.join(descent, on='rbp', rsuffix='_lin')

def _linearity(v):
    if v > 1.:
        return '1+'
    elif v > .7:
        return '0.7-1'
    elif v > 0:
        return '0.7-'
    else:
        return "NA"

def _topR(v):
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

def _domain(dom):
    if type(dom) == float:
        return 'NA'
    from collections import defaultdict
    ds = defaultdict(int)
    for d in dom.split(','):
        ds[d] += 1

    tbl = np.array(sorted([(v,k) for k,v in ds.items()]))[::-1]
    if len(ds.keys()) == 1:
        return ds.keys()[0]
        # return "{} {}".format(ds.values()[0], ds.keys()[0]) # only one domain type
    else:
        return "mixed"
    # return "+".join(sorted(ds.keys()))
    # return tbl[0][1]

def _domain_count(dom):
    if type(dom) == float:
        return 'NA'
    n = len(dom.split(','))
    if n < 4:
        return str(n)
    else:
        return "4+"

def load_descent_run(fname, run='full'):
# descent = pd.read_table('descent2.txt', header=None, names=['rbp','rerr','ferr','corr','steps'])
# descent = pd.read_table('descent_bugfix_noopt.txt', header=None, names=['rbp','rerr','ferr','corr','steps'])
    descent = pd.read_table(fname, header=None, names=['rbp','rerr','ferr','corr','steps'])
    descent.set_index('rbp')
    df = descent.merge(linscore, how='left')
    df = df.merge(domains, how='left')
    df = df.merge(topR, how='left')
    df['motif_linearity'] = df['linscore'].apply(_linearity)
    df['domain'] = df['domains'].apply(_domain)
    df['n_dom'] = df['domains'].apply(_domain_count)
    df['max_R'] = df['top_R'].apply(_topR)
    df['run'] = run
    df.set_index('rbp')

    return df

def label_point(row):
    x,y,val = row
    plt.text(x, y-0.02, str(val))

df = load_descent_run(sys.argv[1])
df_nostruct = load_descent_run(sys.argv[2], run='nostruct')
combined = df.append(df_nostruct)

# print df
def by_R_value_plot(df):
    order = ['20+','5-20','2-5','1-2']
    ax = sns.boxplot(x='max_R', y="corr", data=df, order=order[::-1], whis=np.inf, width=.5, hue='run', dodge=True, hue_order=['nostruct', 'full'])
    ax = sns.swarmplot(x='max_R', y="corr", data=df, order=order[::-1], color=".2", hue='run', dodge=True, hue_order=['nostruct', 'full'])
    plt.xlabel('max. observed 6-mer R-value')
    plt.ylabel('max. R-value correlation after fit')
    plt.ylim(0,1)
    plt.savefig('overview.pdf')
    plt.close()

def by_domain_plot(df):
    # order = ['RRM','RRM+KH','RRM+ZNF', 'RRM+other', 'KH', 'KH+ZNF','KH+other', 'ZNF', 'ZNF+other','other','NA']
    # order = ['1 RRM', '2 RRM', '3 RRM', '4 RRM', '1 KH', '2 KH', '3 KH', '4 KH', '1 ZNF', '2 ZNF', '3 ZNF', '4 ZNF', '1 other', 'mixed']
    order = ['RRM', 'KH', 'ZNF', 'other', 'mixed']
    ax = sns.boxplot(x='domain', y="corr", data=df, order=order, whis=np.inf, width=.5, hue='run', hue_order=['nostruct', 'full'])
    ax = sns.swarmplot(x='domain', y="corr", data=df, order=order, color=".2", hue='run', dodge=True, hue_order=['nostruct', 'full'])

    # lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', hue_order=,legend=True)
    plt.xlabel('RBD type')
    plt.ylabel('final 6mer correlation')
    plt.savefig('fit_by_domain_type.pdf')
    plt.close()

    order = ['1', '2', '3', '4+']
    ax = sns.boxplot(x='n_dom', y="corr", data=df, order=order, whis=np.inf, width=.5, hue='run', hue_order=['nostruct', 'full'])
    ax = sns.swarmplot(x='n_dom', y="corr", data=df, order=order, color=".2", hue='run', dodge=True, hue_order=['nostruct', 'full'])

    # lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', hue_order=,legend=True)
    plt.xlabel('RBD number')
    plt.ylabel('final 6mer correlation')
    plt.savefig('fit_by_domain_num.pdf')
    plt.close()


def corr_scatter(full, nostruct):
    df = full.join(nostruct, lsuffix="_full", rsuffix='_nostruct')

    order = ['20+','5-20','2-5','1-2']
    lmp = sns.lmplot(x='corr_nostruct',y='corr_full',data=df, fit_reg=False, hue='max_R_full', hue_order=order[::-1], legend=True,palette='viridis')
    plt.xlabel('max. Pearson-R nostruct model')
    plt.xlim(0.4,1)
    plt.ylim(0.4,1)
    plt.plot([0,1],[0,1],'-k', linewidth=.1)
    plt.ylabel('max. Pearson-R full model')
    # tolabel.apply(label_point, axis=1)
    plt.savefig('overview_scatter.pdf')
    plt.close()


corr_scatter(df, df_nostruct)
by_R_value_plot(combined)
by_domain_plot(combined)

print "worst proteins using nostruct"
df = df_nostruct.sort_values('corr')
print df[:10]
# rbps = ['RBFOX3','RBFOX2','ELAVL4', 'HNRNPA0','GST','SRSF11','SRSF5','SRSF4','SRSF2','GST','PRR3','ESRP1', 'MSI1']
# rows = df.loc[df['rbp'].isin(rbps)]
# tolabel = rows[['top_R','corr','rbp']]
# corr = df['corr']
# qrange = np.percentile(corr,[25,75])
# corrmean = np.mean(corr)
# print "final best R-value correlation quartile range and mean", qrange, corrmean
# lmp = sns.lmplot(x='top_R',y='corr',data=df, fit_reg=False, hue='motif_linearity', hue_order=['1+','0.7-1','0.7-','NA'],legend=True,palette='viridis')
# # lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='domain', legend=True)
# lmp.set(xscale="log")
# tolabel.apply(label_point, axis=1)

# plt.xlabel('highest 6mer R-value')
# plt.ylabel('final 6mer correlation')
# plt.savefig('overview_linearity.pdf')

# plt.close()


# lmp = sns.lmplot(x='rerr',y='corr',data=df, fit_reg=False, hue='max_R', hue_order=order,legend=True, palette='husl')
# tolabel.apply(label_point, axis=1)

# plt.xlabel('final relative 6mer error')
# plt.ylabel('final 6mer correlation')
# plt.savefig('overview_max_R.pdf')
# plt.close()



