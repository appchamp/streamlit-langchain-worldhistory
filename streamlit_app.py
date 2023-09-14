import os

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import streamlit as st

# read the files
loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader, show_progress=True)
documents = loader.load_and_split()
st.status(f"Loaded {len(documents)} pages..", state="complete")

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
st.status(f"Split docs into total {len(texts)} chunks..", state="complete")

if st.secrets.has_key("openai_key"):
    api_key = st.secrets.openai_key
    st.status("Using existing API Key from st.secrets", state="complete")
    os.environ["OPENAI_API_KEY"] = api_key

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'chroma_db'
embedding = OpenAIEmbeddings()

if st.button("Create Index"):
    with st.status("Creating index..", expanded=True, state="running") as status:
        vectordb = Chroma.from_documents(documents=texts, 
                                         embedding=embedding,
                                         persist_directory=persist_directory)
        vectordb.persist()
        status.update(label="Index creation DONE !", expanded=False, state="complete")

vectordb = None
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()
temp = st.slider(label="Temp", min_value=0.0, max_value=1.0, value=0.0)
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature = temp), 
                                                       chain_type = "stuff",
                                                       retriever=retriever, 
                                                       return_source_documents=True)


## Cite sources
def process_llm_response(llm_response):
    st.markdown(llm_response['result'])  
    st.markdown('\n\nSources:')
    for source in llm_response["source_documents"]:
        st.text(source.metadata['source'])

default_query = '''Act as if you are preparing questions for AP World history exam. Create a question on the Ming Dynasties of the AP World history. The question should be focused on the East Asia region. The question should make students demonstrate understanding the major themes and content areas covered within a given unit. The question should require students to demonstrate their understanding of the material through analysis and interpretation, rather than simply recalling facts. Assign a difficulty level for the question, where the difficulty can vary from 1 to 10 and return it in the "difficulty" parameter.'''

query = st.text_area("Query", value=default_query)
if st.button("Run Query"):
    llm_response = qa_chain(query)
    process_llm_response(llm_response)