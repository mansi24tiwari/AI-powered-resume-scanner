import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('data/UpdatedResumeDataSet.csv')

# Clean the resume text
def cleanResume(txt):
    cleanTxt = re.sub('http\s+\s', ' ', txt)
    cleanTxt = re.sub('RT|CC', ' ', cleanTxt)
    cleanTxt = re.sub('@\s+', ' ', cleanTxt)
    cleanTxt = re.sub('#\s+', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    return cleanTxt

df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Encode the target variable
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Save the LabelEncoder for later use
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))

# Transform the resume text to numerical features
tfidf = TfidfVectorizer(stop_words='english')
requredTaxt = tfidf.fit_transform(df['Resume'])

# Save the TF-IDF vectorizer
pickle.dump(tfidf, open('models/tfidf.pkl', 'wb'))

# Split the data
x_train, x_test, y_train, y_test = train_test_split(requredTaxt, df['Category'], test_size=0.2, random_state=42)

# Train the model
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(x_train, y_train)

# Evaluate the model
ypred = clf.predict(x_test)
accuracy = accuracy_score(y_test, ypred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
pickle.dump(clf, open('models/clf.pkl', 'wb'))
