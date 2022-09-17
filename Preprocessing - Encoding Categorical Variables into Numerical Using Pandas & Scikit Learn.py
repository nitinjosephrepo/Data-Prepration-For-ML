#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns


# In[2]:


df1 = sns.load_dataset('titanic')
#Lets start off with Loading Titanic dataset from seaborn


# In[3]:


df1.head()


# In[4]:


df2 = df1.copy()
#Create a copy of our dataframe 


# ## Encoding Using Pandas

# ### Encoding Features embark_town & class Using Dummies 

# In[5]:


df1.isna().sum()


# In[6]:


dummy_encoded = pd.get_dummies(df1, columns = ["embark_town","class"], prefix = ["town","class"])
#by default get_dummies handles na or missing values. If data requires a separte column for missing values pass If dummy_na = True


# In[7]:


dummy_encoded.head()
#output our Dataframe


# ### Label Encoding using Pandas
# Label encoding is also known as integer encoding. Integer encoding replaces categorical values with numeric values. Here, the unique values in variables are replaced with a sequence of integer values

# In[8]:


class_map = {'class':{'Third':1, 'Second':2, 'First':3},
            'sex':{'male':0, 'female':1}}


# In[9]:


df1 = df1.replace(class_map)


# In[10]:


df1.head()


# ### Ordinal Encoding using Pandas 
# Ordinal encoding is similar to label encoding, except there's an order to the encoding. Ordinal data has ranking. We can define the order of the values as a list and pass it to the category parameter. Lets encode Class

# In[11]:


df1.dtypes


# In[12]:


df1['embark_town'] = df1['embark_town'].astype('category')


# In[13]:


df1.dtypes


# In[14]:


df1['embark_town'] = df1['embark_town'].cat.codes


# In[15]:


df1.head()


# # Encoding Using Scikit Learn

# ### Label Encoding with Sckit Learn

# In[16]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[17]:


df2.head(5)


# In[18]:


labelencoder = LabelEncoder()


# In[19]:


df2['embarked'] = labelencoder.fit_transform(df2['embarked'])


# In[20]:


df2.head()


# ### Onehotencoder with Sklearn with Feature Names 

# In[21]:


df2.dtypes


# In[22]:


df2.isna().sum()


# In[23]:


df2['embark_town'] = df2['embark_town'].fillna(df2['embark_town'].mode()[0])
#we fill missing values in 'embark_town' with most common town where travellers boarded Titanic


# In[24]:


on_hot = OneHotEncoder(sparse=False)
#with sparse = False we make sure that sckit learn returns dense matrix 


# In[25]:


hot_encoded = on_hot.fit_transform(df2[['embark_town']])


# In[26]:


column_names = on_hot.get_feature_names_out(['embark_town'])
#we are able to conveniently access feature names from get_feature_names_out 


# In[27]:


encoded_df = pd.DataFrame(hot_encoded, columns=column_names)


# In[28]:


df2.join(encoded_df)


# ## Ordinal Encoder with Scikit 

# In[29]:


from sklearn.preprocessing import OrdinalEncoder


# In[30]:


Ordinal_encoded = OrdinalEncoder()


# In[31]:


df2[['class']] = Ordinal_encoded.fit_transform(df2[['class']])


# In[32]:


df2.head()


# ## Encoding with Scikit column transformer 

# Preserve Column Order after Sckit Column Transformer

# In[33]:


from sklearn.compose import ColumnTransformer


# In[34]:


col_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), ['embark_town','who'])],
    remainder='passthrough',n_jobs=-1)


# In[35]:


pd.DataFrame(col_transformer.fit_transform(df2), columns=col_transformer.get_feature_names_out()).head()


# ## MakeColumnTransformer

# In[41]:


from sklearn.compose import make_column_transformer


# In[37]:


df2.head()


# In[38]:


transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown = 'ignore'),['sex','embark_town']),
    remainder = 'passthrough',n_jobs=-1)


# In[39]:


transformed = transformer.fit_transform(df2)


# In[40]:


pd.DataFrame(transformed, columns = transformer.get_feature_names_out())


# References: 
#     
# 1. https://stackoverflow.com/questions/68874492/preserve-column-order-after-applying-sklearn-compose-columntransformer/70526434#70526434
# 2. https://towardsdatascience.com/using-columntransformer-to-combine-data-processing-steps-af383f7d5260
# 3. https://pbpython.com/categorical-encoding.html
# 4. https://scikit-learn.ru/example/column-transformer-with-mixed-types/
# 5. https://towardsdatascience.com/categorical-feature-encoding-547707acf4e5
# 6. https://sparkbyexamples.com/pandas/pandas-concat-dataframes-explained/
# 7. https://stackoverflow.com/questions/54570947/feature-names-from-onehotencoder
# 8. https://stackoverflow.com/questions/56338847/how-to-give-column-names-after-one-hot-encoding-with-sklearn
# 9. https://stackoverflow.com/questions/56502864/using-ordinalencoder-to-transform-categorical-values
# 10.https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline.html
# 11.https://towardsdatascience.com/using-columntransformer-to-combine-data-processing-steps-af383f7d5260
# 

# In[ ]:




