import os

from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .assistant import CrimeKGAI


def build_app(verbose: bool) -> CrimeKGAI:
    load_dotenv()

    required_env = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEY"]
    missing = [key for key in required_env if not os.getenv(key)]
    if missing:
        raise SystemExit(f"Missing required env vars in .env: {', '.join(missing)}")

    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    memory = ConversationBufferMemory(return_messages=True)
    index_name = os.getenv("NEO4J_VECTOR_INDEX", "crime_embeddings")

    return CrimeKGAI(
        graph=graph,
        llm=llm,
        embeddings=embeddings,
        vector_index_name=index_name,
        memory=memory,
        verbose=verbose,
    )


def run_validation_test_cases(app: CrimeKGAI) -> None:
    print("\nRunning 3 validation test cases...\n")

    cases = [
        (
            "Reversed relationship direction",
            "Which locations are in area A1?",
            "MATCH (a:Area {areaCode:'A1'})-[:LOCATION_IN_AREA]->(l:Location) RETURN l.address LIMIT 5",
        ),
        (
            "Non-existent label",
            "Show me all suspects.",
            "MATCH (s:Suspect) RETURN s LIMIT 5",
        ),
        (
            "Ambiguous question guessed wrong relationship",
            "Show me connections between people.",
            "MATCH (p:Person)-[:CONNECTED_TO]->(q:Person) RETURN p.name, q.name LIMIT 25",
        ),
    ]

    for title, question, cypher in cases:
        print(f"== {title} ==")
        result = app._validate_cypher(question=question, cypher=cypher)
        print("score:", result.score)
        if result.issues:
            print("issues:")
            for issue in result.issues:
                print("-", issue)
        if result.corrected_cypher:
            print("corrected_cypher:")
            print(result.corrected_cypher)
        print()


def run_demo(app: CrimeKGAI) -> None:
    questions = [
        "Who investigated crimes in area A1?",
        "List crimes that occurred at a given location address.",
        "Recommend crimes similar to burglary.",
        "Find crimes similar to burglary but only those investigated by a specific officer.",
        "Which person has the most connections in this database and who are their top connections?",
    ]
    for question in questions:
        print("\nQ:", question)
        try:
            print("A:", app.answer(question))
        except Exception as e:
            print("A: error:", e)


def run_repl(app: CrimeKGAI) -> None:
    print("Crime KG Assistant (type 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            print(app.answer(question))
        except Exception as e:
            print(str(e))