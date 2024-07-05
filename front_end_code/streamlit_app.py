import streamlit as st
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import docx2txt
import pandas as pd

def extract_text_from_pdf():
    pdf_path = "TRM Guidelines 18 January 2021.pdf"
    text = extract_text(pdf_path)
    return text

def evalutate_ats_score(file_texts, ioc_document):
    texts = [file_texts, ioc_document]
    vector = TfidfVectorizer()
    count_matrix = vector.fit_transform(texts)
    match = cosine_similarity(count_matrix)[0][1]
    match = round(match * 100, 2)
    return match

st.title("💬 Document Evaluation")
st.caption("🚀 A Streamlit chatbot AI specially designed for document evaluation purposes")

with st.sidebar:
    file_format = ("pdf", "doc", "docx", "txt")
    files = st.file_uploader("Upload files", type=file_format, accept_multiple_files=True)

if files:
    results = []
    reference_text = extract_text_from_pdf()

    if 'checkbox_state' not in st.session_state:
        st.session_state['checkbox_state'] = {}

    for file in files:
        if file.type == "application/pdf":
            text = extract_text(file)
        elif file.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
            file_bytes = file.read()
            try:
                text = docx2txt.process(BytesIO(file_bytes))
            except Exception as e:
                st.error(f"Error processing DOCX file: {e}")
                continue
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
        else:
            st.error("Unsupported file format.")
            continue

        ats_score = evalutate_ats_score(reference_text, text)

        if ats_score >= 60:
            checkbox_label = f"{ats_score}% ✔️"
            checkbox_state_key = f"{file.name}_checkbox"
            if checkbox_state_key not in st.session_state['checkbox_state']:
                st.session_state['checkbox_state'][checkbox_state_key] = False
            checkbox = st.checkbox(checkbox_label, value=st.session_state['checkbox_state'][checkbox_state_key], key=checkbox_state_key)
            st.session_state['checkbox_state'][checkbox_state_key] = checkbox
            evidence_text = text[:100] + "..." if checkbox else "N/A"
        else:
            checkbox_label = f"{ats_score}% ❌"
            evidence_text = "N/A"
        
        results.append({
            "Clause": file.name,
            "Full Filled": checkbox_label,
            "Evidence": evidence_text
        })
    df = pd.DataFrame(results)    
    st.subheader("ATS Scores")
    st.table(df)
