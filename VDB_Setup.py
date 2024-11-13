#!/usr/bin/env python
# coding: utf-8

# # Vector Database Collection & API Setup
# 
# This notebook will walk you through setting up the vector database portion of the [openai-realtime-rag](https://github.com/ALucek/openai-realtime-rag/tree/main) fork.

# ## Setting Up Your Vector Database
# 
# For our vector database, a classic choice I use is [ChromaDB](https://www.trychroma.com/). While you can host Chroma as a server itself, I've decoupled the database and the API to allow for more dynamic plug and play capabilities for databases.

# #### Instantiate ChromaDB
# 
# Create a persistent client of ChromaDB that will store everything in the folder `chroma`

# In[1]:


import chromadb

# Creating Vector Database
client = chromadb.PersistentClient()


# In[2]:


client.delete_collection(name="vdb_collection")


# #### Create a New Collection
# 
# This is where all of our chunked text documents are going to be inserted into

# In[3]:


collection = client.get_or_create_collection(name="vdb_collection", metadata={"hnsw:space": "cosine"})


# #### Load & Split PDF 
# 
# We'll be using some simple LangChain integrations to load and chunk our PDF. Using OpenAI's standard token chunk size and overlap for their Assistants API as a baseline.

# In[4]:


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loading and Chunking
loader = PyMuPDFLoader("./Rivian-Case-GPT.pdf")
pages = loader.load()

#loader = PyPDFDirectoryLoader("./PDFs/")
#pages = loader.load()

document = ""
for i in range(len(pages)):
    document += pages[i].page_content

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=800,
    chunk_overlap=400,
)

chunks = text_splitter.split_text(document)


# In[5]:


len(chunks)


# #### Insert Chunks into VDB Collection
# 
# Embed each chunk into the collection

# In[6]:


# Insert Chunks into ChromaDB Collection
i = 0
for chunk in chunks:
    collection.add(
    documents=[chunk],
    ids=[f"chunk_{i}"]
    )
    i += 1


# ---
# ## API Setup
# 
# We'll be using [FastAPI](https://fastapi.tiangolo.com/) as a quick and easy way to host our query function as a REST API. This API is what will be called from the defined `query_db` tool in the main console file.

# In[7]:


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a request model
class QueryRequest(BaseModel):
    query: str

# Define the query endpoint
@app.post("/query")
async def query_chroma(request: QueryRequest):
    # Perform the query on your ChromaDB collection
    results = collection.query(query_texts=[request.query], n_results=5)
    return {"results": results['documents'][0]}


# #### Run API
# 
# Using uvicorn to host the API as a local web server

# In[8]:


import uvicorn
import threading

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the FastAPI app in a background thread
thread = threading.Thread(target=run_api)
thread.start()


# In[ ]:




