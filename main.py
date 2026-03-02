import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Initialize the Neo4j Graph Connection
# LangChain will automatically fetch your Neo4j schema when this runs
try:
    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"]
    )
    print("Successfully connected to Neo4j!")
    
    # Optional: Print the schema to verify it loaded
    print("\nDatabase Schema:")
    print(graph.schema)
    
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")

# 3. Initialize the Gemini Language Model
# We use gemini-1.5-pro or gemini-1.5-flash depending on your preference/access
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0, # Keep temperature at 0 for more deterministic Cypher generation
        google_api_key=os.environ["GEMINI_API_KEY"]
    )
    print("\nSuccessfully initialized Gemini!")
except Exception as e:
    print(f"Error initializing Gemini: {e}")