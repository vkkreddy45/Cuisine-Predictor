import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

def Readdata(idata):
    df=pd.read_json(idata)
    return df

def preprocess(df,ip):
    #exclude = set(string.punctuation)
    stop_words = set(stopwords.words("english"))
    inglist=[]
    x=[]
    for i in df['ingredients']:
        i = " ".join(i)
        x.append(i)
#     i = " ".join([i for i in df['ingredients']])
#     x.append(i)
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

def vectorization(data):
    #print(data)
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(data)
    headings = vector[0]
    vector = vector[1:]
    print("Done with Vectorization!")
    #print(vector)
    return headings, vector

def model(headings, vector, df):
    # kernel = 'rbf' = 77.43
    # sigmoid = 65
    # poly = 66.3
    # SVM() = 77.64,77.78
    print("Starting Model Buliding!")
    LabelEncoder = preprocessing.LabelEncoder()
    LabelEncoder.fit(df['cuisine'])
    clf = make_pipeline(SVC())
    print("Created Pipeline!")
    X = vector
    Y = LabelEncoder.transform(df['cuisine'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    print("Done with Split, Train and Test!!")
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    try:
        print("Model Accuracy :",accuracy_score(y_test, y_pred) * 100)
    except:
        print("Value is Invalid")
    inputPredict = clf.predict(headings)
    cuisine = LabelEncoder.inverse_transform(inputPredict)
    print("Cuisine: ",cuisine)

def TopnRecipe(headings, vector, df, N):
    df['Scores'] = cosine_similarity(headings,vector).transpose()
    #df['Scores'] = scores
    closestRecipe = df[['id','Scores']].nlargest(N, ['Scores'])
    print("Closest 10 Recipes \n",closestRecipe)

def start(ip,N):
    df = "yummly.json"
    df = Readdata(df)
    data = preprocess(df,ip)
    headings, vector = vectorization(data)
    model(headings, vector, df)
    TopnRecipe(headings, vector, df, N)

#ip = ["chili powder", "crushed red pepper flakes", "garlic powder", "sea salt", "ground cumin", "onion powder", "dried oregano", "ground black pepper", "paprika"]
#df = "yummly.json"
#df = Readdata(df)
#data = preprocess(df,ip)
#headings, vector = vectorization(data)
#model(headings, vector, df)
#TopnRecipe(headings, vector, df, N)
