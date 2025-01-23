#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv('UpdatedResumeDataSet.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[11]:


df['Category'].value_counts()


# In[16]:


print(df['Category'].dtype)


# In[18]:


df['Category'] = df['Category'].astype('category')
import seaborn as sns
sns.countplot(x='Category', data=df)


# In[25]:


# Convert the 'Category' column to a categorical type
df['Category'] = df['Category'].astype('category')

# Now plot
plt.figure(figsize=(15, 5))
sns.countplot(x=df['Category'])
plt.xticks(rotation=90)
plt.show()


# In[29]:


df['Category'].unique()


# In[37]:


counts =df['Category'].value_counts()
labels =df['Category'].unique()
plt.figure(figsize=(15, 10))

plt.pie(counts,labels=labels,autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()


# In[38]:


df['Category'][0]


# In[39]:


df['Resume']


# In[45]:


import re
def cleanResume(txt):
    cleanTxt = re.sub('http\s+\s',' ',txt)
    cleanTxt = re.sub('RT|CC',' ',cleanTxt)
    cleanTxt = re.sub('@\s+',' ',cleanTxt)
    cleanTxt = re.sub('# \s+',' ',cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
    cleanTxt = re.sub('\s+',' ',cleanTxt)                  
    return cleanTxt


# In[46]:


df['Resume'] = df['Resume'].apply(lambda x:cleanResume(x))


# In[47]:


df['Resume'][0]


# In[50]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[51]:


le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])


# In[52]:


print(df['Category'])


# In[53]:


df.Category.unique()


# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf =TfidfVectorizer(stop_words='english')
    
tfidf.fit(df['Resume'])
requredTaxt = tfidf.transform(df['Resume'])


# In[60]:


##splitting
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(requredTaxt, df['Category'], test_size=0.2, random_state=42)



# In[61]:


x_train.shape


# In[62]:


x_test.shape


# In[67]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(x_train, y_train)
ypred = clf.predict(x_test)
print(accuracy_score(y_test,ypred))


# In[70]:


##prediction system
import pickle
pickle.dump(tfidf,open('tfidf.pkl','wb'))
pickle.dump(clf, open('clf.pkl','wb'))


# In[72]:


myresume = """Mansi Tiwari
+91 7987201846	  mansitiwari1224@gmail.com	  https://www.linkedin.com/in/mansi-tiwari-8a082a220/
  https://github.com/mansi24tiwari
Education
Amity University Raipur Chhattisgarh	2021-2025
B Tech CSE	Raipur, Chhattisgarh
Experience
Atharv Optikos Syndsseis Private Limited — Full Stack Developer Intern
10 July 2023 – 30 Aug 2023 | Raipur, Chhattisgarh
•	Automated deployment pipelines using CI/CD tools, reducing deployment time by 40%.
•	Designed and developed a scalable web application using React, Node.js, and MongoDB, improving user experience for 500+ active users.
•	Implemented a caching mechanism that boosted query response efficiency by 30%.
•	Collaborated with a cross-functional team to optimize database performance and enhance backend workflows.
Teachnook & Internshala — Machine Learning Intern
01 July 2023 – 5 Sept 2023
•	Gained hands-on experience in data preprocessing, model development, and evaluation techniques.
•	Built and optimized machine learning models for various educational use cases, applying techniques to handle imbalanced data and improve model accuracy.
•	Collaborated with mentors to deploy ML solutions, gaining a deep understanding of real-world challenges in AI-driven applications.
•	Enhanced technical skills in Python, scikit-learn, and TensorFlow while working on domain-specific projects.
Projects
      1. AI-Powered Resume Scanner
Technologies: Python, NLP, TensorFlow, Django
•	Developed an ATS-compatible resume parser using TensorFlow and NLP, achieving 95% accuracy in parsing over 1,000+ resumes.
•	Automated candidate screening, reducing manual workload by 40%.
2. Critico: Modular Course Management System
Technologies: HTML, CSS, JavaScript, Flask, React
•	Engineered a course management system for faculty-student collaboration, improving productivity for 200+ users.
•	Deployed RESTful APIs with JWT-based authentication to enhance data security by 20%.
•	Hosted on AWS, achieving 99.9% uptime.
3. Distributed System Simulation
Technologies: Kubernetes, Docker, Distributed Systems
•	Designed a distributed file-sharing system leveraging fault-tolerant data storage principles.
•	Deployed on a cloud-based Kubernetes cluster for seamless scalability.
Technical Skills
Languages: English, Hindi, beginners French.
Technologies: Git, GitHub, TypeScript, C++, HTML, CSS, JavaScript, python, java, NLP, TensorFlow, Django, Flask, ReactJS, NodeJS, MongoDB, NextJS, SQL, Docker, Kubernetes, PostgreSQL, Redis, AWS.
Concepts: Computer Networks, Operating System, Artificial Intelligence, Machine Learning, Neural Networks, Distributed Systems, Scalability, System Design, or Data Structures and Algorithms.
"""


# In[77]:


import pickle

clf = pickle.load(open('clf.pkl','rb'))
cleaned_resume = cleanResume(myresume)
input_features = tfidf.transform([cleaned_resume])
prediction_id =clf.predict(input_features)[0]
category_mapping ={
    15:"Java Developer",
    23:"Testing",
    8:"DeveOps Engineer",
    20:"Python Developer",
    24:"Web Designing",
    12:"HR",
    13:"Hadoop",
    3:"Blockchain",
    10:"ETL Developer",
    18:"Operations Manager",
    6:"Data Science",
    22:"Sales",
    16:"Mechanical Engineer",
    1:"Arts",
    7:"Database",
    11:"Sales",
    14:"Health and fitness",
    19:"PMO",
    4:"BUsiness Analyst",
    9:"DotNet Developer",
    2:"Automation Testing",
    17:"Network Security Engineer",
    21:"SAP Developer",
    5:"Civil Engineer",
    0:"Advocate"
}
category_name = category_mapping.get(prediction_id, "Unknown")
print ("Predicted Category:", category_name)
print(prediction_id)


# In[ ]:




