from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
#Loading API Keys
load_dotenv()
#load document
loader=PyPDFLoader("Poem.pdf")
docs=loader.load()
#splitting
splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
chunks=splitter.split_documents(docs)
#embeddings
embeddings=OpenAIEmbeddings()
#vector DB
db=FAISS.from_documents(chunks,embeddings)
#Retriever
retriever=db.as_retriever()
# LLM
llm = ChatOpenAI()
#parser
parser=StrOutputParser()
#Prompt
prompt=PromptTemplate(
    template=
    """
You are a helpful AI Assistant
Please answer the {query} based on the given {context}

""",
input_variables=["context","query"]
)
#Ask
query=input("Please Enter what you gonna ask:")
retrieved_docs=retriever.invoke(query)
# Convert to text
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
#Building the chain
chain=prompt|llm|parser
print(chain.invoke({"context":context,"query":query}))

