# IMPORTS
from category_encoders.target_encoder import TargetEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# DATASET
dataset = pd.read_csv('Churn_Modelling.csv')
# Encoding
te = TargetEncoder()
dataset[['Geography', 'Gender']] = te.fit_transform(
    dataset[['Geography', 'Gender']], dataset.Exited)
# Columns of interest
df = dataset[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
              'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]

# VARIABLES TO DISPLAY
# Orginal
org_head = dataset.head()
# Filtered
fil_head = df.head()
fil_stats = df.describe()

# STREAMLIT
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title('View/Criteria')
options = st.sidebar.selectbox(
    'Select', ['Dataset', 'Total Exited', 'Correlation between columns', 'Percentage values', 'Shorter Tenure', 'Number of Products', 'Age'])
# Dataset

if options == 'Dataset':
    st.title('Why are customers leaving the bank? - EDA')
    st.write('## Dataset')
    st.write('First 5 rows :')
    st.write(org_head)
    st.write('Columns of interest :')
    st.write(fil_head)
    st.write('Statistics of the filtered dataset :')
    st.write(fil_stats)
    st.write('''### Formulating Questions

    * Find out the percentage of people who left the bank when their salary was less than average.
    * Did people who have shorter tenure leave the bank?
    * What percentage of people with higher balance leave?
    * Did number of products affect their decision to leave?
    * Does age affect decision to leave the bank?
''')
# Plot 1
if options == 'Total Exited':
    st.title('Total number of people who left the bank')
    plt.figure(figsize=(8, 6))
    sns.set_style('dark')
    sns.countplot(x='Exited', data=df, palette='magma')
    st.pyplot()
# Improving correlation
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1
df2 = df[~((df.iloc[:, :-1] < (q1 - 1.5 * iqr)) |
           (df.iloc[:, :-1] > (q3 + 1.5 * iqr))).any(axis=1)].copy()
df2.dropna(inplace=True)
# Plot 2
if options == 'Correlation between columns':
    st.title('Correlation between columns')
    st.write('### Correlation')
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), linecolor='white', linewidths=3, annot=True)
    st.pyplot()
    st.write('Although most attributes dont have high correlation, it seems like age and geography slightly affect whether people leave the bank or not.')
    st.write('### Correlation after removing outliers')
    plt.figure(figsize=(12, 10))
    sns.heatmap(df2.corr(), linecolor='white', linewidths=3, annot=True)
    st.pyplot()
    st.write('### Before vs After')
    fig, ax = plt.subplots(1, 2, figsize=(34, 12))
    sns.heatmap(df[['Age', 'Geography', 'Exited']].corr(
    ), linecolor='white', linewidths=3, annot=True, ax=ax[0], cmap='YlOrRd', annot_kws={"size": 24})
    sns.heatmap(df2[['Age', 'Geography', 'Exited']].corr(
    ), linecolor='white', linewidths=3, annot=True, ax=ax[1], cmap='PuBuGn', annot_kws={"size": 24})
    st.pyplot()
    st.write(
        'Although it is not a drastic change, the correlation between Age and Exited has improved.')
# Percentage
if options == 'Percentage values':
    st.write(
        'Percentage of people who left the bank when their salary was less than average')
    avg_sal = df.EstimatedSalary.mean()
    below_avg = df.loc[(df.EstimatedSalary < avg_sal) & (df.Exited == 1)]
    st.write(below_avg.Exited.count()/100, '%')
    st.write('What percentage of people with higher balance leave?')
    st.write(print(df.loc[(df.Balance > df.Balance.mean())
             & (df.Exited == 1)].Balance.count()/100, '%'))

# Plot 3
if options == 'Shorter Tenure':
    st.title('Did people who have shorter tenure leave the bank?')
    plt.figure(figsize=(10, 8))
    sns.countplot(df.Tenure, hue='Exited', data=df, palette='inferno')
    st.pyplot()
    st.write('### Conclusion')
    st.write(
        'Looks like tenure does not affect whether customers leave the bank or not.')
# Plot 4
if options == 'Number of Products':
    st.title('Did number of products affect their decision to leave?')
    products_ex = df[['NumOfProducts', 'Exited']]
    products_ex = products_ex.loc[(products_ex.Exited == 1)]
    group = products_ex.groupby('NumOfProducts', as_index=False).count()
    counts = []
    counts.append(df.loc[(df.NumOfProducts == 1)].NumOfProducts.count())
    counts.append(df.loc[(df.NumOfProducts == 2)].NumOfProducts.count())
    counts.append(df.loc[(df.NumOfProducts == 3)].NumOfProducts.count())
    counts.append(df.loc[(df.NumOfProducts == 4)].NumOfProducts.count())
    for i in range(len(counts)):
        st.write('Total percentage of people with '+str(i+1) +
                 ' product(s) :', (counts[i]/100), '%')
    st.write('## People Exited based on Number of Products')
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Exited', y='NumOfProducts',
                data=group, orient='h', palette='winter')
    st.pyplot()
    st.write('### Dataframe')
    group
    st.write('### Conclusion')
    st.write('''From the plot we can observe that almost all the people with NumOfProducts=[3, 4] left the bank, but Since the total number of samples for ordinals 3 and 4 is very less compared to 1 and 2, it cannot be concluded that people with more NumOfProducts leave the bank.
             ''')
# Plot 5
if options == 'Age':
    st.title('Does age affect decision to leave the bank?')
    plt.figure(figsize=(10, 8))
    sns.set_style('darkgrid')
    sns.boxplot(x="Exited", y="Age", data=df, palette='viridis')
    st.pyplot()
    st.write('## After removing outliers')
    plt.figure(figsize=(10, 8))
    sns.set_style('darkgrid')
    sns.boxplot(x="Exited", y="Age", data=df2, palette='cool')
    st.pyplot()
    st.write('### Conclusion')
    st.write('The boxplot tells that majority of the people who left the bank are around 40-55 years old. ')
