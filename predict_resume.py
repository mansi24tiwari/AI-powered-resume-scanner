import pickle
import re

# Load the saved models
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
clf = pickle.load(open('models/clf.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))

# Function to clean the resume text
def cleanResume(txt):
    cleanTxt = re.sub('http\s+\s', ' ', txt)
    cleanTxt = re.sub('RT|CC', ' ', cleanTxt)
    cleanTxt = re.sub('@\s+', ' ', cleanTxt)
    cleanTxt = re.sub('#\s+', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    return cleanTxt

# Define a new resume for prediction
myresume = """Mansi Tiwari
+91 7987201846	  mansitiwari1224@gmail.com	  https://www.linkedin.com/in/mansi-tiwari-8a082a220/
  https://github.com/mansi24tiwari
Education
Amity University Raipur Chhattisgarh	2021-2025
B Tech CSE	Raipur, Chhattisgarh
Experience
Atharv Optikos Syndsseis Private Limited — Full Stack Developer Intern
10 July 2023 – 30 Aug 2023 | Raipur, Chhattisgarh
...
"""

# Clean and transform the resume text
cleaned_resume = cleanResume(myresume)
input_features = tfidf.transform([cleaned_resume])

# Predict the category
predicted_category_id = clf.predict(input_features)[0]
predicted_category_name = le.inverse_transform([predicted_category_id])[0]

# Print the prediction
print("Predicted Category:", predicted_category_name)
