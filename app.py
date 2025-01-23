import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('models/clf.pkl','rb'))
tfidf =pickle.load(open('models/tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_lines = []
    for line in resume_text.splitlines():
        if not re.search(r'(RT|CC|http|\@|\#)', line):
            clean_lines.append(line)
    clean_text = "\n".join(clean_lines)
    clean_text = re.sub('http\s+\s', ' ', resume_text)
    clean_text = re.sub('RT|CC', ' ', clean_text)
    clean_text = re.sub('@\s+', ' ', clean_text)
    clean_text = re.sub('# \s+', ' ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
#web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['pdf','txt'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            if uploaded_file.name.endswith('.pdf'):
                import fitz
                pdf_document = fitz.open(stream=resume_bytes, filetype="pdf")
                resume_text = ""
                for page in pdf_document:
                    resume_text += page.get_text("text")
            else:
                resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(f"Prediction ID: {prediction_id}")

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DeveOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Sales",
            14: "Health and fitness",
            19: "PMO",
            4: "BUsiness Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
        }
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)

#python main
if __name__== "__main__":
    main()