import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      if not chunks:
            st.write("No text chunks were extracted from the PDF.")
            return
      
      # create embeddings
      embeddings = OllamaEmbeddings(base_url=os.getenv("OLLAMA_HOST"), model="nomic-embed-text")
      
      if not embeddings:
          st.write("No embeddings found.")
          return

      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = Ollama(base_url=os.getenv("OLLAMA_HOST"), model="llama3")
        chain = load_qa_chain(llm, chain_type="stuff")
        
        response = chain.run(input_documents=docs, question=user_question)        
           
        st.write(response)    

if __name__ == '__main__':
    main()