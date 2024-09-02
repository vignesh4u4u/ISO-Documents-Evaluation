from llama_cpp import Llama
import streamlit as st
import pandas as pd
import fitz
import tempfile
import re
import os

from prompt_agent import (evidence_prompt_recheck,
                          evidence_context,
                          rationale_prompt,
                          finalCheck_rational_evidence,
                          evidence_prompt1)

st.set_page_config(page_title="CISO.AI", page_icon=":rocket:")

st.markdown("""
    <style>
    .top-logo {
        display: flex;
        align-items: center;
        justify-content: left;
        padding: 5px; 
        background-color: #f8f9fa; 
    }
    .top-logo h1 {
        margin: 0;
        font-size: 24px;
        color: #007bff; 
    }
    .emoji {
        font-size: 32px; 
        margin-right: 10px;
    }
    </style>
    <div class="top-logo">
        <span class="emoji">🤖</span>
        <h1>Document Evaluation with CISO.AI</h1>
    </div>
    """, unsafe_allow_html=True)


if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'show_files' not in st.session_state:
    st.session_state.show_files = False

if 'show_dataframe' not in st.session_state:
    st.session_state.show_dataframe = False

if 'filter_options' not in st.session_state:
    st.session_state.filter_options = []

if 'process_running' not in st.session_state:
    st.session_state.process_running = False

if 'data' not in st.session_state:
    st.session_state.data = []


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


class Extractor:
    @staticmethod
    def extract_decision(mapping):
        mapping_match = re.search(r'No|Yes', mapping, re.IGNORECASE)
        return mapping_match.group(0) if mapping_match else None

    @staticmethod
    def extract_rationale(mapping):
        rationale_match = re.search(r'Rationale:\s*(.*)', mapping.strip(), re.IGNORECASE | re.DOTALL)
        return rationale_match.group(1).strip() if rationale_match else None


class StreamlitFileprocess:
    def process_uploaded_files(self, files):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_list = []
            for file in files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                pdf_document = fitz.open(file_path)
                text_content = ""
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    text = page.get_text("text")
                    text_content += text + "\n"
                pdf_document.close()
                data_list.append({"file_name": file.name, "text": text_content.strip()})
            data_df = pd.DataFrame(data_list)
            return data_df

    def generate_initial_results(self, df, data1):
        results = []
        for index_df, row_df in df.iterrows():
            clause = row_df['Clause']
            clause_no = row_df["Clause_Number"]
            for index_data1, row_data1 in data1.iterrows():
                cleaned_text = row_data1['cleaned_text']
                prompt = evidence_prompt1(cleaned_text, clause)
                results.append({
                    "Clause_Number": clause_no,
                    'Clause': clause,
                    'File_name': row_data1['file_name'],
                    'Cleaned_text': cleaned_text,
                    "Evidence_initial_prompt": prompt
                })
        return pd.DataFrame(results)


class LlamaCppInterface:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(current_dir, "../Model/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf")
        self.model_config = {
            "model_path": self.model_path,
            "n_gpu_layers": 100,
            "n_ctx": 9000,
            "seed": 42,
            "temperature": 0.1,
            "max_tokens": 1000,
            "verbose": False
        }

    def main_llama_cpp(self, query):
        llm = Llama(**self.model_config)
        response = llm.create_chat_completion(messages=[{"role": "user", "content": query}], stream=True)
        return response


llama_interface = LlamaCppInterface()
file_processor = StreamlitFileprocess()
uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
else:
    st.session_state.uploaded_files = []

start_button_color = "lightblue" if st.session_state.uploaded_files else "skyblue"
dataframe_button_color = "lightblue" if st.session_state.show_dataframe else "skyblue"
show_files_button_color = "lightblue" if st.session_state.show_files else "skyblue"


table_container = st.empty()

if 'data' not in st.session_state:
    st.session_state.data = []

def update_table(clause_no, file_name, symbol,evidence):
    if st.session_state.data is not None:
        st.session_state.data.append({
            "clause_no": clause_no,
            "file_name": file_name,
            "Final_Mapping_Decision": symbol,
            "Evidence":evidence
        })
        table_container.write(pd.DataFrame(st.session_state.data))


st.sidebar.markdown(f"""
    <style>
    /* DataFrame Button */
    [data-testid="stSidebar"] .stButton > button:first-child {{
        background-color: {dataframe_button_color};
        color: white;
        margin-bottom: 30px
    }}
    /* Show Files Button */
    [data-testid="stSidebar"] .stButton > button:nth-child(2) {{
        background-color: {show_files_button_color};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)


if st.button('🚀 Start Process'):
    if st.session_state.uploaded_files:
        st.session_state.show_files = False
        st.session_state.show_dataframe = False
        st.session_state.process_running = True  # Set process as running

        results_list = []
        data1 = file_processor.process_uploaded_files(st.session_state.uploaded_files)
        data1['cleaned_text'] = data1['text'].apply(clean_text)
        data1 = data1.drop(columns=['text'])
        iso_clauses_df = pd.read_excel("ISO_Clauses_Processed.xlsx")
        pattern = r'(\d+\.\d+)'
        iso_clauses_df['Clause_Number'] = iso_clauses_df['Clause'].str.extract(pattern)
        iso_clauses_df = iso_clauses_df[['Clause_Number'] + [col for col in iso_clauses_df.columns if col != 'Clause_Number']]
        initial_results_df = file_processor.generate_initial_results(iso_clauses_df.iloc[0:1], data1)
        st.session_state.combined_df = initial_results_df
        data = pd.DataFrame(st.session_state.combined_df)

        st.session_state.results_df = data
        results_list = []
        for index, row_df in data.iterrows():
            clause = row_df['Clause']
            cleaned_text = row_df['Cleaned_text']
            file_name = row_df['File_name']
            evidence_initial_prompt = row_df["Evidence_initial_prompt"]
            clause_no = row_df["Clause_Number"]

            placeholder = st.empty()
            response_container = st.empty()
            response_container_recheck = st.empty()

            response = ""
            for chunk in llama_interface.main_llama_cpp(evidence_initial_prompt):
                chunk_message = chunk['choices'][0]['delta'].get('content', '')
                if chunk_message:
                    response += chunk_message
                    #response_container.markdown(f"{response}")
            initial_response = response

            recheck_prompt = evidence_prompt_recheck(cleaned_text, clause, initial_response)

            response_recheck = ""
            for chunk in llama_interface.main_llama_cpp(recheck_prompt):
                chunk_message = chunk['choices'][0]['delta'].get('content', '')
                if chunk_message:
                    response_recheck += chunk_message
                    #response_container_recheck.markdown(f"{response_recheck}")
            initial_response_recheck = response_recheck
            decision = Extractor.extract_decision(initial_response_recheck)

            if decision and decision.lower() == "yes":
                evidence_context_prompt = evidence_context(cleaned_text, clause)
                response_container_context = st.empty()
                response_context = ""
                for chunk in llama_interface.main_llama_cpp(evidence_context_prompt):
                    chunk_message = chunk['choices'][0]['delta'].get('content', '')
                    if chunk_message:
                        response_context += chunk_message
                        #response_container_context.markdown(f"{response_context}")
                evidence = response_context

                rationale_prompt_context = rationale_prompt(clause, cleaned_text, evidence)
                response_container_rationale = st.empty()
                response_rationale = ""
                for chunk in llama_interface.main_llama_cpp(rationale_prompt_context):
                    chunk_message = chunk['choices'][0]['delta'].get('content', '')
                    if chunk_message:
                        response_rationale += chunk_message
                        #response_container_rationale.markdown(f"{response_rationale}")
                rationale = response_rationale

                final_mapping_prompt = finalCheck_rational_evidence(clause, evidence, rationale)
                response_container_final = st.empty()
                response_final = ""
                for chunk in llama_interface.main_llama_cpp(final_mapping_prompt):
                    chunk_message = chunk['choices'][0]['delta'].get('content', '')
                    if chunk_message:
                        response_final += chunk_message
                        #response_container_final.markdown(f"{response_final}")
                rationale_final = response_final
                decision = Extractor.extract_decision(rationale_final)
                if decision.lower() == "yes":
                    symbol = "✅"
                    update_table(clause_no, file_name, symbol,evidence)

            else:
                evidence = initial_response_recheck
                rationale = initial_response_recheck
                rationale_final = initial_response_recheck

                decision = Extractor.extract_decision(rationale_final)
                if decision.lower() == "no":
                    symbol = "❌"
                    evidence= None
                    update_table(clause_no, file_name, symbol,evidence)


            print(f"Row {index} - Initial response: {initial_response}")
            print(f"Row {index} - Recheck response: {initial_response_recheck}")
            print(f"Row {index} - Context response: {evidence}")
            print(f"Row {index} - Rationale response: {rationale}")
            print(f"Row {index} - Final rationale: {rationale_final}")

            results_list.append({
                "Clause_Number": clause_no,
                'Clause': clause,
                'File_name': file_name,
                'Document_text': cleaned_text,
                'Evidence': evidence,
                'Rationale': rationale,
                'Final_mapping': rationale_final
            })

        st.session_state.final_df = pd.DataFrame(results_list)

        st.session_state.process_running = False
        st.write("✅ Document analysis is complete.")
    else:
        st.write("⚠️ Please upload files first.")


if st.sidebar.button('📊 DataFrame'):
    if not st.session_state.process_running:
        st.session_state.show_dataframe = True
        st.session_state.show_files = False


if st.sidebar.button('📂 Show Files'):
    if not st.session_state.process_running:
        st.session_state.show_files = True
        st.session_state.show_dataframe = False


if st.sidebar.button('🧹 Clear'):
    if not st.session_state.process_running:
        # Clear all session state variables related to uploaded files and outputs
        st.session_state.uploaded_files = []
        st.session_state.show_files = False
        st.session_state.show_dataframe = False
        st.session_state.filter_options = []
        st.session_state.data = []
        st.session_state.combined_df = pd.DataFrame()
        st.session_state.final_df = pd.DataFrame()  # Clear final DataFrame
        st.write("Cleared all outputs!")

if st.session_state.show_files and st.session_state.uploaded_files:
    st.write("### 📂 Uploaded Files:")
    for file in st.session_state.uploaded_files:
        st.write(f"- {file.name}")

if st.session_state.show_dataframe:
    st.write("### 📊 DataFrame View:")
    if 'final_df' in st.session_state:
        st.dataframe(st.session_state.final_df)

        csv = st.session_state.final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )
    else:
        st.write("No DataFrames to display.")

st.markdown("""
    <style>
    .stApp {
        padding: 10px; 
        margin-left: 0; 
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("---")
st.write("[Privacy Policy](#) | [Contact Info](#)")
