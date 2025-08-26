from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
name = os.environ.get("PINECONE_INDEX", "memories")
if name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=os.environ.get("PINECONE_CLOUD","aws"),
                            region=os.environ.get("PINECONE_REGION","us-east-1"))
    )
    print("Created index", name)
else:
    print("Index exists", name)
