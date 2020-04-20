from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd
import string
import nltk
import re

# Read data & pepreaparing it by tockenizing, steming words, and removing punctuation. Adding length of abstractas a feature 

data = pd.read_csv("", sep='\t')
data.columns = ['Usefulness', 'abstract']

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data['abstract_len'] = data['abstract'].apply(lambda x: len(x) - x.count(" "))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
    
    #spliting data to train and test
X_train, X_test, y_train, y_test = train_test_split(data[['abstract', 'abstract_len']], data['Usefulness'], test_size=0.2)


#Vectoring text to encode text 
tfidfvect = TfidfVectorizer(analyzer=clean_text)
tfidfvectfit = tfidfvect.fit(X_train['abstract'])

tfidfvectfit_train = tfidfvectfit.transform(X_train['abstract'])
tfidfvectfit_test = tfidfvectfit.transform(X_test['abstract'])

X_train_vect = pd.concat([X_train[['abstract_len']].reset_index(drop=True), 
           pd.DataFrame(tfidfvectfit_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['abstract_len']].reset_index(drop=True), 
           pd.DataFrame(tfidfvectfit_test.toarray())], axis=1)

X_train_vect.head()



#Using svc classifier for prediction
svc=SVC()
svc_model = svc.fit(X_train_vect, y_train)
svc_model.score(X_test_vect,y_test)


#Using GradientBoosting classifier for prediction
#Using GradientBoosting classifier for prediction
MNB = MultinomialNB()
MNB_model=MNB.fit(X_train_vect, y_train)
MNB_model.score(X_test_vect,y_test)
                  

