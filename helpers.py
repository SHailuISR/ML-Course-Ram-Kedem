import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def show_hist(col,title=None):
  text =  'Stats:' + '\n\n'
  text +=  'Mean: ' + str(round(col.mean(), 2)) + '\n'
  text += 'Median: ' + str(round(col.median(), 2)) + '\n'
#   text += 'Mode: ' + str(list(col.mode().values)[0]) + '\n'
  text += 'Std dev: ' + str(round(col.std(), 2)) + '\n'
  text += 'Skew: ' + str(round(col.skew(), 2)) + '\n'

  bn = round(col.count() ** (1/3)) *2

  col.plot(kind='hist', bins = bn)
  plt.axvline(col.mean(), color='k', linestyle='dashed', linewidth=1)
  plt.axvline(col.median(), color='red', linestyle='dashed', linewidth=1)
  plt.text(0.95, 0.45, text, fontsize=12, transform=plt.gcf().transFigure)
  plt.title(title, fontsize=16, fontweight="bold");

def show_box(col):
    text =  'Stats:' + '\n\n'
    text +=  'mean: ' + str(round(col.mean(), 2)) + '\n'
    text +=  'quantile 25: ' + str(round(col.quantile(.25), 2)) + '\n'
    text += 'quantile 50: ' + str(round(col.quantile(.50), 2)) + '\n'
    text += 'quantile 75: ' + str(round(col.quantile(.75), 2)) + '\n'
    text += 'iqr: ' + str(round(col.quantile(.75)-col.quantile(.25), 2)) + '\n'
    plt.text(0.95, 0.55, text, fontsize=12, transform=plt.gcf().transFigure)
    col.plot(kind='box', vert=False)
    
    plt.show()

def show_counts(column1, column2=None):
    ax = sns.countplot(x = column1, hue=column2);
    for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.15, p.get_height()), ha='center', va='top', color='white', size=12)

def show_bar(df):
    ax = df.plot(kind='bar');
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(),2), (p.get_x()+0.15, p.get_height()), ha='center', va='top', color='white', size=10)

def get_numeric_details(df, sort_column='mean', sort_order=False):
    res = pd.DataFrame()
    numeric_columns = df.select_dtypes(include='number').columns

    for column in numeric_columns:
        data = pd.DataFrame({'min':[df[column].min()],
                             'quantile 25':df[column].quantile(.25),
                             'quantile 50':df[column].quantile(.50),
                             'quantile 75':df[column].quantile(.75),
                             'max':df[column].max(),
                             'mean':df[column].mean(),
                             'median':df[column].median(),
                             'mode': ','.join(str(obj) for obj in list(df[column].mode().values)),
                             'std':df[column].std(),
                             'count':df[column].count(),
                             'nunique':df[column].nunique(),
                             'skew':df[column].skew()
                            },index=[column])
        res = res.append(data)
    return res

def ci_unknown_std(sample, alpha):
    sample_size = sample.size
    sample_mean = sample.mean()
    t_critical  = stats.t.ppf(q = 1-alpha/2, df=sample.size-1)  
    sample_stdev = sample.std(ddof=1) 
    sigma = sample_stdev/math.sqrt(sample.size) 
    margin_of_error = t_critical * sigma
    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)  
    return confidence_interval
    
def show_distribution(df,col):
  #creating two subplots:
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
  textb = 'Quantiles:\n\n 25%: ' + str(round(df[col].quantile(.25), 2)) + '\n 50%: ' + str(round(df[col].quantile(.50), 2)) + '\n 75%: ' + str(round(df[col].quantile(.75), 2)) + '\n'
  plt.text(0.8, 0.65, textb, fontsize=12, transform=plt.gcf().transFigure)
  #prepare text:
  text = 'Stats:\n\nMean: ' + str(round(df[col].mean(), 2)) + '\nMedian: ' + str(round(df[col].median(), 2)) + '\nStd dev: ' + str(round(df[col].std(), 2)) + '\n'
  plt.text(0.35, 0.65, text, fontsize=12, transform=plt.gcf().transFigure)
    # histplot:        
  sns.histplot(x=df[col],ax=axes[0])
  axes[0].set_title('Histogram')
  #boxplot:
  sns.boxplot(ax=axes[1],x=df[col],showmeans=True,meanline=True,meanprops={'color':'white'})
  axes[1].set_title('Box-Plot')

  #main title:
  fig.suptitle(col.capitalize() +' distribution', fontsize=20,fontweight="bold")

def calc_anova(df, group_column, values_column):
    # Get list of unique group values in provided column
    unique_group_values = df[group_column].drop_duplicates().to_list()

    # Iterate through each unique group value and filter the dataframe to get the values
    # of the provided column for that group, then store them in a list
    values_by_group = []
    for group_value in unique_group_values:
        group_filter = df[group_column] == group_value
        values_by_group.append(df[values_column][group_filter])

    # Perform ANOVA test on the list of value arrays using the `f_oneway` function from the `scipy.stats` module
    return f_oneway(*values_by_group)