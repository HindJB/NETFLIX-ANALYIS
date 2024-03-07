#!/usr/bin/env python
# coding: utf-8

# In[244]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[245]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[246]:


netflix=pd.read_csv('Desktop/8. Netflix Dataset.csv')


# In[247]:


netflix.head()


# In[248]:


netflix.tail()


# In[249]:


netflix.describe()


# In[250]:


netflix.info()


# In[251]:


netflix.shape


# In[252]:


netflix.columns


# In[253]:


netflix.isnull().sum()


# In[254]:


netflix.nunique()


# In[255]:


netflix.duplicated().sum()


# In[256]:


df = netflix.copy()


# In[257]:


df.shape


# In[258]:


df=df.dropna()
df.shape


# In[259]:


df.head(10)


# In[260]:


df["Release_Date"] = pd.to_datetime(df['Release_Date'])
df['day_added'] = df['Release_Date'].dt.day
df['year_added'] = df['Release_Date'].dt.year
df['month_added']=df['Release_Date'].dt.month
df['year_added'].astype(int);
df['day_added'].astype(int);


# In[261]:


df.head(10)


# In[262]:


bar_colors = ['orangered', 'black']
sns.countplot(x='Category', data=netflix, palette=bar_colors)

fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Category')


# In[263]:


bar_colors = sns.dark_palette("orangered", n_colors=14, reverse=True)
sns.countplot(x='Rating',palette=bar_colors,  data=netflix, order=netflix['Rating'].value_counts().index)

plt.xticks(rotation=90, ha='right')

fig = plt.gcf()
fig.set_size_inches(13, 13)

plt.title('Rating')

plt.show()


# In[264]:


bar_colors = ['orangered', 'black']
plt.figure(figsize=(10,8))
sns.countplot(x='Rating',hue='Category',data=netflix, palette=bar_colors)
plt.title('Relation between Category and Rating')
plt.show()


# In[265]:


bar_colors = ['orangered', 'black']
labels = ['Movie', 'TV show']
size = netflix['Category'].value_counts()
colors = bar_colors
explode = [0, 0.1]
plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, labels=labels, colors=colors, explode=explode, shadow=True, startangle=90)
plt.title('Distribution of Type', fontsize=25)
plt.legend()
plt.show()


# In[266]:


custom_palette = sns.color_palette(['orangered', 'black'])

netflix['Rating'].value_counts().plot.pie(
    autopct='%1.1f%%',
    shadow=True,
    figsize=(10, 8),
    colors=custom_palette
)
plt.title('Rating')

plt.show()


# In[267]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud


# In[268]:


def color_func(*args, **kwargs):
    return 'orangered'

plt.subplots(figsize=(25, 15))
wordcloud = WordCloud(
    background_color='white',
    width=1920,
    height=1080,
    color_func=color_func
).generate(" ".join(df.Country))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('Country.png')
plt.show()


# In[269]:


def color_func(*args, **kwargs):
    return 'orangered'

plt.subplots(figsize=(25, 15))
wordcloud = WordCloud(
    background_color='white',
    width=1920,
    height=1080,
    color_func=color_func
).generate(" ".join(df.Director))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('Director.png')
plt.show()


# In[271]:


def color_func(*args, **kwargs):
    return 'orangered'

plt.subplots(figsize=(25, 15))
wordcloud = WordCloud(
    background_color='white',
    width=1920,
    height=1080,
    color_func=color_func
).generate(" ".join(df.Type))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('Type.png')
plt.show()


# In[272]:


netflix["Director"].value_counts().head(10) 


# In[273]:


netflix_top_10_dates = netflix.sort_values(by='Release_Date').head(10)[['Title', 'Release_Date']]
print(netflix_top_10_dates)


# In[277]:


netflix_top_10_dates['Release_Date'] = pd.to_datetime(netflix_top_10_dates['Release_Date'])
custom_palette = sns.color_palette(['orangered', 'black'])


plt.figure(figsize=(10, 6))
plt.bar(netflix_top_10_dates['Title'], netflix_top_10_dates['Release_Date'], color=custom_palette)

plt.xlabel('Title')
plt.ylabel('Release Date')
plt.title('Top 10 Movies and TV Shows Release Dates')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




