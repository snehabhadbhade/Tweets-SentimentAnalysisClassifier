import re
from sklearn.naive_bayes import BernoulliNB
import sys
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
#from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
#from skearn.vm import LinearSVC

stemmer = PorterStemmer()
def tokenizer(text):
	#remove non letter chars
	#text = re.sub("[^a-zA-Z]","",text)
	#nltk tokenize
	tokens = word_tokenize(text)
	#stem
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

train_file = sys.argv[1]
test_file = sys.argv[2]

with open(train_file,"r") as f:
	train_text = f.readlines()
f.close()

with open(test_file,"r") as f:
        test_text = f.readlines()
f.close()

train_data = []
train_target = np.ndarray(shape = len(train_text), dtype ='int64')
pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
labels = []

for line in train_text:
	data = line.split(",")
	tweet = data[5]
	tweet = tweet.lower()
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
	tweet = re.sub('@[^\s]+','',tweet)
	tweet = re.sub('[\s]+', ' ', tweet)
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	tweet = tweet.strip('\'"')
	tweet = pattern.sub(r"\1\1",tweet)
	tweet = tweet.strip('\'"?,.:;!-_')
	tweet = re.sub("[^a-zA-Z]"," ",tweet)
	train_data.append(tweet)
	labels.append(int(data[0].strip('"')))

#print labels
train_target = np.asarray(labels, dtype='int64')
test_data = []
test_target = []
for line in test_text:
        data = line.split(",")
        tweet = data[4]
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
        tweet = re.sub('@[^\s]+','',tweet)
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = tweet.strip('\'"')
        tweet = pattern.sub(r"\1\1",tweet)
        tweet = tweet.strip('\'"?,.')
        tweet = re.sub("[^a-zA-Z]"," ",tweet)
	test_data.append(tweet)
        #train_labels.append(data[0])

print("training started")

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),}

linear_pipe = Pipeline([('vect', CountVectorizer(encoding="latin-1",ngram_range=(1,2))),('tfidf',TfidfTransformer(use_idf=False)), ('clf', MultinomialNB()),])

#gs_svm = GridSearchCV(linear_pipe, parameters, n_jobs=-1)
gs_svm = linear_pipe.fit(train_data, train_target)

#vectorizer = TfidfVectorizer(encoding="latin-1", analyzer='word',stop_words='english',min_df=3,ngram_range=(1,3))
#X_train = vectorizer.fit_transform(train_data)
#clf = MultinomialNB().fit(X_train,train_target)
#Y_train = vectorizer.transform(test_data)

print("training completed")
#predicted = clf.predict(Y_train)
predicted = gs_svm.predict(test_data)

print("writing to file")

with open("output.csv", "w") as f:
	count=0
	for label in predicted:
		text = test_text[count].split(",")
		id = text[0].strip('"')
		id.rstrip()
		row = id+","+str(label)+"\n"
		f.write(row)
		count+=1
f.close()

