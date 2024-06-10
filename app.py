import streamlit as st
from inference_pipeline import LLM_RAG

def main():
    st.title("LLM RAG Inference App")

    query = st.text_input("Enter your query:")
    enable_rag = st.checkbox("Enable RAG", value=False)
    enable_evaluation = st.checkbox("Enable Evaluation", value=False)
    enable_monitoring = st.checkbox("Enable Monitoring", value=True)
    uploaded_file = st.file_uploader("Upload a file for update Vector DB (Qdrant)", type=["pdf", "csv", "xlsx", "docx", "mp3"])

    if st.button("Submit"):
        if query:
            model = LLM_RAG()
            result = model.generate(
                query=query,
                enable_rag=enable_rag,
                enable_evaluation=enable_evaluation,
                enable_monitoring=enable_monitoring,
            )
            st.write("Answer:", result["answer"])
            if result["llm_evaluation_result"]:
                st.write("Evaluation Result:", result["llm_evaluation_result"])
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()