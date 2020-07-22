#!/usr/bin/env python
# coding: utf-8

# **MODULI**

# In[134]:


import pandas as pd
import numpy as np
import re
import csv

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from nltk.corpus import stopwords as sw
from nltk.stem.snowball import SnowballStemmer
from num2words import num2words
import spacy

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# **DATA EXPLORATION**

# In[2]:


d_types={"text":str,"class":str}
df=pd.read_csv("development.csv",dtype=d_types)

d_types={"text":str}
df2=pd.read_csv("evaluation.csv",dtype=d_types)


# In[3]:


print(df.info())
print()
print(df2.info())


# In[4]:


print(df.isna().any(axis=0))
print()
print(df2.isna().any(axis=0))


# *ok, no missing values*

# In[5]:


df["text"][0]


# 
# *analyzing a random review it's possible to see there are many grammar errors and some senteces don't make any sense. This could be due to the fact that these reviews was originally in another language and then were translate to italian. Or a simpler explanation is that a customer could be done many mistakes if he writes fastly*
# 

# In[6]:


set(df["class"])


# In[7]:


number_pos=df[df["class"]=="pos"].shape[0]
number_neg=df[df["class"]=="neg"].shape[0]

print(number_pos)
print(number_neg)


# In[8]:


x=np.arange(2)
y=[number_pos,number_neg]

fig,ax=plt.subplots(figsize=(10,8))
ax.set_title("Reviews",fontsize=30)
ax.set_ylabel("Number of reviews",fontsize=20)
plt.bar(x,y,color=["green","red"])
plt.xticks(x,("positive reviews","negative reviews"),fontsize=20)
for i in range(len(x)):
    ax.annotate(y[i],(x[i],y[i]),xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=15)
plt.savefig("number_reviews.png")
plt.show()


# *so there are much more positive reviews than negative. For this reason when the train_test split will be performed it mandatory to use a stratify approach
# in order to mantain this proprortion*

# **PREPROCESSING**

# In[9]:


def number_conversion(word):
    try:
        conv=num2words(int(word),lang="it")
    except:
        conv=""
    return conv

lemmatizer = spacy.load("it_core_news_sm",disable=["ner","parser","tagger"])
stem=SnowballStemmer("italian")

def pre_process(row):
    
    #CASE NORMALIZATION - PUNCTUACTION ELIMINATION - EMOJI ELIMINATION
    re.compile("[.;:,\"()\[\]]").sub("",row.lower()) 
    re.compile("(\n)|\'|(\-)|(\/)").sub(" ",row)
    row.encode('ascii', 'ignore').decode('ascii')
        
    #NUMBER CONVERSION
    res=" ".join([number_conversion(word) if word.isnumeric() else word for word in row.split()])
    
    #STOPWORD ELIMINATION
    stop_words=sw.words('italian')
    if "non" in stop_words:
        stop_words.remove("non")
    res_1 =" ".join(["" if word in stop_words else word for word in res.split()])

    #SHORT AND LONG WORDS ELIMINATION
    res_2=" ".join(["" if len(word)<3 else word for word in res_1.split()])

    #LEMMATIZER
    par=lemmatizer(res_2)
    risul=" ".join([token.lemma_ for token in par])

    #STEMMING
    final=" ".join([stem.stem(word)for word in risul.split()])
    
    #STOPWORD ELIMINATION
    stop_words=sw.words('italian')
    if "non" in stop_words:
        stop_words.remove("non")
    finale_2 =" ".join(["" if word in stop_words else word for word in final.split()])
    
    return finale_2

def label_transform(label):
    lista=[]
    for lab in label:
        if lab=="neg":
            lista.append(0)
        else:
            lista.append(1)
    return lista


# **TRAIN-TESTING**

# In[10]:


df_train,df_test=train_test_split(df,train_size=0.8,shuffle=True,stratify=df["class"])


# In[64]:


reviews_train=df_train["text"].apply(pre_process)
labels_train=label_transform(df_train["class"])
reviews_test=df_test["text"].apply(pre_process)
labels_test=label_transform(df_test["class"])


# *TF-IDF*

# In[12]:


tfidf_vectorizer=TfidfVectorizer(binary=False,ngram_range=(1,3),min_df=5,max_df=0.7)
tfidf_vectorizer.fit(reviews_train)
x_train_tfidf=tfidf_vectorizer.transform(reviews_train)
x_test_tfidf=tfidf_vectorizer.transform(reviews_test)


# *SVD*

# In[13]:


svd_tfidf=TruncatedSVD(n_components=10000)
svd_tfidf.fit_transform(x_train_tfidf)
print(f"Total variance explained: {np.sum(svd_tfidf.explained_variance_ratio_):.2f}")


# In[15]:


cum_variance = np.cumsum(svd_tfidf.explained_variance_ratio_)
idx1 = np.argmax(cum_variance > .7)
value=cum_variance[idx1]

x_plot=np.linspace(1,len(cum_variance),len(cum_variance))

fig,ax=plt.subplots(figsize=(10,8))
ax.set_title("Explained variance",fontsize=30)
ax.set_ylabel("Explained variance ratio",fontsize=10)
ax.set_xlabel("Number of components",fontsize=10)
ax.plot(x_plot,cum_variance)
plt.scatter(idx1+1,value)
string=f"{idx1} components"
ax.annotate(string,(idx1,value),xytext=(-20,10),textcoords="offset points",ha="center",va="bottom",fontsize=15)
plt.savefig("reviews_train.png")
plt.show()


# In[16]:


svd_tfidf=TruncatedSVD(n_components=6503)
svd_tfidf.fit_transform(x_train_tfidf)
print(f"Total variance explained: {np.sum(svd_tfidf.explained_variance_ratio_):.2f}")

x_train_transform_tfidf =svd_tfidf.transform(x_train_tfidf)
x_test_transform_tfidf = svd_tfidf.transform(x_test_tfidf)


# *understand if data are linear*

# In[17]:


plt.scatter(x_train_transform_tfidf[:,0],labels_train)
plt.show()
plt.scatter(x_train_transform_tfidf[:,1],labels_train)
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor=LinearRegression().fit(x_train_transform_tfidf,labels_train)
print(r2_score(regressor.predict(x_train_transform_tfidf),labels_train))

svd_fake=TruncatedSVD(n_components=2)
svd_fake.fit(x_train_tfidf)

x_train_fake =svd_fake.transform(x_train_tfidf)
x_test_fake = svd_fake.transform(x_test_tfidf)

plt.scatter(x_train_fake[:,0],x_train_fake[:,1],c=labels_train)
plt.show()


# *Linear SVC*

# In[18]:


param_grid={"penalty":["l2"],"loss":["hinge","squared_hinge"],"dual":[True],
            "tol":[0.001,0.01,0.1,1,2,5],"C":[0.001,0.01,0.1,1,2,5],"random_state":[42],"max_iter":[1000,2000]}
grid_search=GridSearchCV(LinearSVC(),param_grid,scoring="f1",cv=5)
grid_search.fit(x_train_transform_tfidf,labels_train)


# In[19]:


print("LINEAR SVC")
print(grid_search.best_estimator_)
print()
print(f"grid search score: {grid_search.best_score_}")
print()
predict_finale_svc=grid_search.predict(x_test_transform_tfidf)
f1=f1_score(labels_test, predict_finale_svc,average="weighted")
print("f1 (test set): %.4f" % f1)


# *Logistic Regressor*

# In[20]:


param_grid_logistic_regression={"penalty":["l2"],"dual":[False],"tol":[0.005,0.01,0.1],"C":[0.1,0.5,1,2,5],
                                "random_state":[42],"solver":["sag","saga"],"max_iter":[50,100,200]}
grid_search_logistic_regression=GridSearchCV(LogisticRegression(),param_grid_logistic_regression,scoring="f1",cv=5)
grid_search_logistic_regression.fit(x_train_transform_tfidf,labels_train)


# In[21]:


print("LOGISTIC REGRESSION")
print(grid_search_logistic_regression.best_estimator_)
print()
print(f"grid search score: {grid_search_logistic_regression.best_score_}")
print()
predict_finale_logistic_regression=grid_search_logistic_regression.predict(x_test_transform_tfidf)
f1=f1_score(labels_test, predict_finale_logistic_regression,average="weighted")
print("f1 (test set): %.4f" % f1)


# *Count*

# In[22]:


count_vectorizer=CountVectorizer(binary=False,ngram_range=(1,3),min_df=5,max_df=0.7)
count_vectorizer.fit(reviews_train)
x_train_count=count_vectorizer.transform(reviews_train)
x_test_count=count_vectorizer.transform(reviews_test)


# *Multinomial NB*

# In[23]:


param_grid_multinomial_nb={"alpha":[0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7],"fit_prior":[True,False]}
grid_search_multinomial_nb=GridSearchCV(MultinomialNB(),param_grid_multinomial_nb,scoring="f1",cv=5)
grid_search_multinomial_nb.fit(x_train_count,labels_train)


# In[24]:


print("MULTINOMIAL NB")
print(grid_search_multinomial_nb.best_estimator_)
print()
print(f"grid search score: {grid_search_multinomial_nb.best_score_}")
print()
predict_finale_multinomial_nb=grid_search_multinomial_nb.predict(x_test_count)
f1=f1_score(labels_test, predict_finale_multinomial_nb,average="weighted")
print("f1 (test set): %.4f" % f1)


# *Final*

# In[61]:


finale=[]
for i in range(len(predict_finale_svc)):
    if predict_finale_svc[i]==predict_finale_logistic_regression[i]:
        finale.append(predict_finale_svc[i])
    else:
        finale.append(predict_finale_multinomial_nb[i])


# In[62]:


f1=f1_score(labels_test, finale,average="weighted")
print("f1 (test set): %.4f" % f1)


# In[213]:


matrix=confusion_matrix(finale,labels_test)
labels = ['neg', 'pos'] 

print(matrix) 

fig,ax = plt.subplots(figsize=(5,5)) 
cm = ax.matshow(matrix,cmap="YlGn") 

ax.set_xticklabels([''] + labels) 

ax.set_yticklabels([''] + labels) 
       
plt.xlabel('Predicted') 

plt.ylabel('Actual') 

ax.text(0,0,matrix[0,0])
ax.text(0,1,matrix[0,1])
ax.text(1,0,matrix[1,0])
ax.text(1,1,matrix[1,1],color="white")

plt.savefig("confusion_matrix.png")

plt.show()


# **MODEL OK, CLASSIFICATION**

# In[214]:


reviews_deve=df["text"].apply(pre_process)
labels_deve=label_transform(df["class"])
reviews_eva=df2["text"].apply(pre_process)


# *TF-IDF*

# In[215]:


tfidf_vectorizer=TfidfVectorizer(binary=False,ngram_range=(1,3),min_df=5,max_df=0.7)
tfidf_vectorizer.fit(reviews_deve)
x_deve_tfidf=tfidf_vectorizer.transform(reviews_deve)
x_eva_tfidf=tfidf_vectorizer.transform(reviews_eva)


# *SVD*

# In[216]:


svd_tfidf_deve=TruncatedSVD(n_components=10000)
svd_tfidf_deve.fit_transform(x_deve_tfidf)
print(f"Total variance explained: {np.sum(svd_tfidf_deve.explained_variance_ratio_):.2f}")


# In[217]:


cum_variance = np.cumsum(svd_tfidf_deve.explained_variance_ratio_)
idx1 = np.argmax(cum_variance > .7)
value=cum_variance[idx1]

x_plot=np.linspace(1,len(cum_variance),len(cum_variance))

fig,ax=plt.subplots(figsize=(10,8))
ax.set_title("Explained variance",fontsize=30)
ax.set_ylabel("Explained variance ratio",fontsize=10)
ax.set_xlabel("Number of components",fontsize=10)
ax.plot(x_plot,cum_variance)
plt.scatter(idx1+1,value)
string=f"{idx1} components"
ax.annotate(string,(idx1,value),xytext=(-20,10),textcoords="offset points",ha="center",va="bottom",fontsize=15)
plt.savefig("reviews_deve.png")
plt.show()


# In[218]:


svd_tfidf=TruncatedSVD(n_components=8065)
svd_tfidf.fit_transform(x_deve_tfidf)
print(f"Total variance explained: {np.sum(svd_tfidf.explained_variance_ratio_):.2f}")

x_deve_transform_tfidf =svd_tfidf.transform(x_deve_tfidf)
x_eva_transform_tfidf = svd_tfidf.transform(x_eva_tfidf)


# *Linear SVC*

# In[38]:


# linear_svc=grid_search.best_estimator_
# linear_svc.fit(x_deve_transform_tfidf,labels_deve)

# predict_finale_svc=linear_svc.predict(x_eva_transform_tfidf)


# In[219]:


param_grid={"penalty":["l2"],"loss":["hinge","squared_hinge"],"dual":[True],
            "tol":[0.001,0.01,0.1,1,2,5],"C":[0.001,0.01,0.1,1,2,5],"random_state":[42],"max_iter":[1000,2000]}
grid_search=GridSearchCV(LinearSVC(),param_grid,scoring="f1",cv=5)
grid_search.fit(x_deve_transform_tfidf,labels_deve)


# In[220]:


print("LINEAR SVC")
print(grid_search.best_estimator_)
print()
print(f"grid search score: {grid_search.best_score_}")
print()
predict_finale_svc=grid_search.predict(x_eva_transform_tfidf)


# *Logistic Regressor*

# In[39]:


# logistic_regression=grid_search_logistic_regression.best_estimator_
# logistic_regression.fit(x_deve_transform_tfidf,labels_deve)

# predict_finale_logistic=logistic_regression.predict(x_eva_transform_tfidf)


# In[221]:


param_grid_logistic_regression={"penalty":["l2"],"dual":[False],"tol":[0.005,0.01,0.1],"C":[0.1,0.5,1,2,5],
                                "random_state":[42],"solver":["sag","saga"],"max_iter":[50,100,200]}
grid_search_logistic_regression=GridSearchCV(LogisticRegression(),param_grid_logistic_regression,scoring="f1",cv=5)
grid_search_logistic_regression.fit(x_deve_transform_tfidf,labels_deve)


# In[222]:


print("LOGISTIC REGRESSION")
print(grid_search_logistic_regression.best_estimator_)
print()
print(f"grid search score: {grid_search_logistic_regression.best_score_}")
print()
predict_finale_logistic_regression=grid_search_logistic_regression.predict(x_eva_transform_tfidf)


# *Count*

# In[223]:


count_vectorizer=CountVectorizer(binary=False,ngram_range=(1,3),min_df=5,max_df=0.7)
count_vectorizer.fit(reviews_deve)
x_deve_count=count_vectorizer.transform(reviews_deve)
x_eva_count=count_vectorizer.transform(reviews_eva)


# *Multinomial NB*

# In[43]:


# multinomial=grid_search_multinomial_nb.best_estimator_
# multinomial.fit(x_deve_count,labels_deve)

# predict_finale_multinomial=multinomial.predict(x_eva_count)


# In[224]:


param_grid_multinomial_nb={"alpha":[0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7],"fit_prior":[True,False]}
grid_search_multinomial_nb=GridSearchCV(MultinomialNB(),param_grid_multinomial_nb,scoring="f1",cv=5)
grid_search_multinomial_nb.fit(x_deve_count,labels_deve)


# In[225]:


print("MULTINOMIAL NB")
print(grid_search_multinomial_nb.best_estimator_)
print()
print(f"grid search score: {grid_search_multinomial_nb.best_score_}")
print()
predict_finale_multinomial_nb=grid_search_multinomial_nb.predict(x_eva_count)


# *Final*

# In[226]:


finale=[]
for i in range(len(predict_finale_svc)):
    if predict_finale_svc[i]==predict_finale_logistic_regression[i]:
        finale.append(predict_finale_svc[i])
    else:
        finale.append(predict_finale_multinomial_nb[i])


# In[235]:


def re_convert_label(label):
    lista=[]
    for lab in label:
        if lab==0:
            lista.append("neg")
        else:
            lista.append("pos")
    return lista

predict=re_convert_label(finale)

with open('exam.csv', 'w',encoding='utf-8') as fp:
    string="Id,Predicted"
    fp.write(f'{string}\n')
    for i in range (len(predict)):
        fp.write(f'{i},{predict[i]}\n')

