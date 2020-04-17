from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import string
import nltk
import re

# Read data & pepreaparing it by tockenizing, steming words, and removing punctuation. Adding length of title as a feature 

data = pd.read_csv("", sep='\t')
data.columns = ['Usefulness', 'title']

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data['title_len'] = data['title'].apply(lambda x: len(x) - x.count(" "))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
    
    #spliting data to train and test
X_train, X_test, y_train, y_test = train_test_split(data[['title', 'title_len']], data['Usefulness'], test_size=0.2)


#Vectoring text to encode text 
tfidfvect = TfidfVectorizer(analyzer=clean_text)
tfidfvectfit = tfidfvect.fit(X_train['title'])

tfidfvectfit_train = tfidfvectfit.transform(X_train['title'])
tfidfvectfit_test = tfidfvectfit.transform(X_test['title'])

X_train_vect = pd.concat([X_train[['title_len']].reset_index(drop=True), 
           pd.DataFrame(tfidfvectfit_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['title_len']].reset_index(drop=True), 
           pd.DataFrame(tfidfvectfit_test.toarray())], axis=1)

X_train_vect.head()

#Using RandomForest classifier for prediction
RandomForest = RandomForestClassifier(n_estimators=100)
RandomForest_model = RandomForest.fit(X_train_vect, y_train)
RandomForest_model.score(X_test_vect,y_test)


#Using GradientBoosting classifier for prediction
GradientBoosting = GradientBoostingClassifier(n_estimators=100)
GradientBoosting_model = GradientBoosting.fit(X_train_vect, y_train)
GradientBoosting_model.score(X_test_vect,y_test)
