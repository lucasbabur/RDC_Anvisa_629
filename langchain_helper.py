from langchain_openai import OpenAI # Diferença é que ChatOpenAI é conversação, OpenAI é LLM.
# from langchain_core.prompts import PromptTemplate # Chat Prompt Template permite você dar as instruções do sistema.
from langchain_core.output_parsers import StrOutputParser # Transforma \n em pulo de linha, formata direitinho, suprime metadados...
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("RDC 629.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = OpenAI()


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def invoke (question):
    return rag_chain.invoke(question)

