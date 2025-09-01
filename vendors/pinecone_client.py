import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX = pc.Index(os.getenv("PINECONE_INDEX", "uap-kb"))
