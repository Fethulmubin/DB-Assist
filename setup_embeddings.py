import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os




load_dotenv()

# 1. Initialize the Gemini Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"]
)
print("Embedding model initialized.")

# 2. Calculate and store embeddings in Neo4j
# This will fetch all 'CrimeInvestigation' nodes, combine their 'title' and 'tagline',
# send them to Gemini to get the 768-dimension vector, and save it to the 'embedding' property.
try:
    vector_store = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        index_name="crime_embeddings",     # The name of the vector index
        node_label="CrimeInvestigation",                # Which nodes to embed
        text_node_properties=["Area", "Crime"], # Properties to turn into text for the embedding
        embedding_node_property="embedding", # Where to store the vector on the node
    )
    print("Successfully generated embeddings and created the vector index!")

except Exception as e:
    print(f"Error setting up embeddings: {e}")
    
#test = vector_store.similarity_search("What crime happened in Area 1?", k=3)
results = vector_store.similarity_search("What crime happened in Area 1?", k=3)
print("Similarity search results:")
for result in results:
    print(f"Node ID: {result['id']}, Properties: {result['properties']}")
    