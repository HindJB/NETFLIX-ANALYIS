#!/usr/bin/env python
# coding: utf-8

# ### Data Loading
# 

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


netflix=pd.read_csv('Desktop/8. Netflix Dataset.csv')


# ### Data Viewing
# #### First 5 Values
# 

# In[5]:


netflix.head()


# #### Last 5 Values

# In[6]:


netflix.tail()


# #### Describe The Data
# 
# 

# In[7]:


netflix.describe()


# #### Information About The Data

# In[8]:


netflix.info()


# In[ ]:





# In[9]:


netflix.shape


# In[10]:


netflix.columns


# In[11]:


netflix.isnull().sum()


# In[12]:


netflix.nunique()


# In[13]:


netflix.duplicated().sum()


# In[14]:


df = netflix.copy()


# In[15]:


df.shape


# In[16]:


df=df.dropna()
df.shape


# In[17]:


df.head(10)


# In[18]:


df["Release_Date"] = pd.to_datetime(df['Release_Date'])
df['day_added'] = df['Release_Date'].dt.day
df['year_added'] = df['Release_Date'].dt.year
df['month_added']=df['Release_Date'].dt.month
df['year_added'].astype(int);
df['day_added'].astype(int);


# In[19]:


df.head(10)


# ### What are the Categories of content available on Netflix?

# In[21]:


import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

types = df['Category'].value_counts().reset_index()

trace = go.Pie(labels=types['Category'], values=types['count'], 
               pull=[0.1, 0], marker=dict(colors=["orangered", "black"]),
               title="Netflix Content Category")
fig = go.Figure([trace])
fig.show()


# We see that most of the content broadcast on Netflix is created by TV shows

# ### What is the number of content added to Netflix by years?

# In[77]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'df' is your DataFrame with columns 'Release_Date' and other relevant columns
# Replace 'Release_Date' with the actual column name in your DataFrame
df['Release_Date'] = pd.to_datetime(df['Release_Date'])
df['year_added'] = df['Release_Date'].dt.year

# Count the number of content additions by year
content_by_year = df['year_added'].value_counts().sort_index()

# Plotting a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=content_by_year.index, y=content_by_year.values, color='orangered')
plt.xlabel('year_added')
plt.ylabel('Number of Content Additions')
plt.title('Content Added to Netflix Over the Years')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# ### Grouped by Country and Category

# In[42]:


show_origin= df.groupby(['Country','Category'])[ 'Category'].count().reset_index(name='show_count')
show_origin


# ### Netflix Content by Ratings

# In[23]:


bar_colors = sns.dark_palette("orangered", n_colors=14, reverse=True)
sns.countplot(x='Rating',palette=bar_colors,  data=netflix, order=netflix['Rating'].value_counts().index)

plt.xticks(rotation=90, ha='right')

fig = plt.gcf()
fig.set_size_inches(13, 13)

plt.title('Rating')

plt.show()


# ### Relation between Category and Rating

# In[24]:


bar_colors = ['orangered', 'black']
plt.figure(figsize=(10,8))
sns.countplot(x='Rating',hue='Category',data=netflix, palette=bar_colors)
plt.title('Relation between Category and Rating')
plt.show()


# ### Content Rating Distribution

# In[25]:


custom_palette = sns.color_palette(['orangered', 'black'])

netflix['Rating'].value_counts().plot.pie(
    autopct='%1.1f%%',
    shadow=True,
    figsize=(10, 8),
    colors=custom_palette
)
plt.title('Rating')

plt.show()


# In[26]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud


# ###  Netflix Content Distribution Around the Globe

# In[27]:


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


# ###  WordCloud of Directors Behind Netflix Productions

# In[28]:


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


# ###  WordCloud of Content Types

# In[29]:


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


# ### Top 10 Directors

# In[30]:


netflix["Director"].value_counts().head(10) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




