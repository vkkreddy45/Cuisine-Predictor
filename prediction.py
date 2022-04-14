# Importing the required set of libraries

import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Reading the Inputed Data

def Readdata(idata):
    df=pd.read_json(idata)
    return df

# Preprocessing the data with Normalization Concepts

def preprocess(df,ip):
    inglist=[]
    x=[]
    stop_words=set(stopwords.words("english"))
    for i in df['ingredients']:
        i = " ".join(i)
        x.append(i)
    ip = " ".join(ip)
    x.insert(0,ip)

    for j in x:
        j = j.lower()
        j = ''.join('' if j.isdigit() else c for c in j)
        token = nltk.word_tokenize(j)
        mytokens = " ".join([word for word in token if word not in stop_words])
        inglist.append(mytokens)
    print("Done with Data Preprocessing!")
    #print(inglist)
    return inglist

# Vectorizing the data

def vectorization(data):
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(data)
    headings = vector[0]
    vector = vector[1:]
    print("Done with Vectorization!")
    #print(vector)
    return headings, vector

# Creating a model and under going the Training, Testing and Spliting

def model(headings, vector, df):
    print("Starting Model Building!")
    LabelEncoder = preprocessing.LabelEncoder()
    LabelEncoder.fit(df['cuisine'])
    clf = make_pipeline(SVC())
    print("Created Pipeline!")
    x_train, x_test, y_train, y_test = train_test_split(vector, LabelEncoder.transform(df['cuisine']), test_size=0.3)
    print("Done with split, Train and Test!!")
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    inputPredict = clf.predict(headings)
    cuisine = LabelEncoder.inverse_transform(inputPredict)
    cusine_dt = {}
    cusine_dt['cuisine'] = cuisine[0]
    # print(cusine_dt)
    return cusine_dt

# Finding the top n Values

def TopnRecipe(headings, vector, df, N):
    df['Scores'] = cosine_similarity(headings,vector).transpose()
    # df['Scores'] = scores
    closestRecipe = df[['id','Scores']].nlargest(int(N)+1, ['Scores'])
    # closestRecipe = closestRecipe.set_index('id')
    closest_recipes = closestRecipe.to_dict('records')

    # print(type(closest_recipes))
    dictionary = closest_recipes[0]
    score = dictionary['Scores']
    # print("Closest 10 Recipes \n",closestRecipe)
    return score,closest_recipes[1:]

# Printing the Final Output in Json Format

def print_final_output(final_dt):
    json_object = json.dumps(final_dt, indent = 4)
    #print(final_dt)
    print(json_object)

# Creating a Json File and Writing the data into it.

def write_to_file(final_dt):
    # print(type(final_dt))
    with open('scores.json', 'w') as json_file:
        json.dump(final_dt, json_file)

def start(ip,N):
    final_dt = {}
    df = "yummly.json"
    df = Readdata(df)
    data = preprocess(df,ip)
    headings, vector = vectorization(data)
    cusine_dt = model(headings, vector, df)
    final_dt.update(cusine_dt)
    score,close_recipes = TopnRecipe(headings, vector, df, N)
    final_dt['score'] = score
    final_dt['closest'] = close_recipes
    print_final_output(final_dt)
    write_to_file(final_dt)
