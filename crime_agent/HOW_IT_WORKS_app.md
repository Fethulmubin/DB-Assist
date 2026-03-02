# HOW IT WORKS — `app.py`

This file is the **application layer** between the raw `CrimeKGAI` class and the user's terminal.
It reads configuration from environment variables, constructs every dependency, and offers three
different runtime modes: a quick demo, a validation test suite, and an interactive REPL.

---

## `build_app(verbose: bool) -> CrimeKGAI`

### Purpose
Construct a fully wired `CrimeKGAI` instance from environment variables without the caller
needing to know anything about how the pieces are assembled.

### How It Works

**Step 1 — Load environment variables**
```python
load_dotenv()
```
`python-dotenv` reads a `.env` file in the project root (if present) and injects each line as an
environment variable. This means developers can store secrets locally in `.env` rather than
setting them on the shell each time.

**Step 2 — Fail fast on missing config**
```python
required_env = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEY"]
missing = [key for key in required_env if not os.getenv(key)]
if missing:
    raise SystemExit(f"Missing required env vars in .env: {', '.join(missing)}")
```
All four variables are mandatory. If any are absent, the program exits immediately with a clear
error message rather than crashing later with a confusing `KeyError` or authentication failure.

**Step 3 — Connect to Neo4j**
```python
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)
```
`Neo4jGraph` is a LangChain wrapper. On construction it connects to the database and
auto-fetches the schema (node labels, relationship types, property keys), making it immediately
available as `graph.schema` for all prompts.

**Step 4 — Create the Gemini LLM**
```python
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    temperature=0,
    google_api_key=os.environ["GEMINI_API_KEY"],
)
```
`temperature=0` is used because all responses must be deterministic and structured (JSON or
Cypher). Creative variation would cause JSON parsing failures. The model name defaults to
`gemini-flash-latest` but can be overridden via the `GEMINI_MODEL` environment variable.

**Step 5 — Create the embedding model**
```python
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    google_api_key=os.environ["GEMINI_API_KEY"],
)
```
The same API key is shared between the chat model and the embedding model. The embedding model
defaults to `gemini-embedding-001` (768 dimensions) which must match the dimension used when the
Neo4j vector index was created.

**Step 6 — Create conversation memory**
```python
memory = ConversationBufferMemory(return_messages=True)
```
`return_messages=True` means the memory's `load_memory_variables` returns a list of
`BaseMessage` objects rather than a concatenated string. This is required by the prompt
templates which use `MessagesPlaceholder("chat_history")`.

**Step 7 — Assemble and return**
```python
index_name = os.getenv("NEO4J_VECTOR_INDEX", "crime_embeddings")
return CrimeKGAI(graph=graph, llm=llm, embeddings=embeddings,
                 vector_index_name=index_name, memory=memory, verbose=verbose)
```

### Inputs
| Parameter | Type   | Description |
|-----------|--------|-------------|
| `verbose` | `bool` | Passed directly to `CrimeKGAI`. When `True`, routing, Cypher, and validation details are printed to stdout. |

### Output
`CrimeKGAI` — a fully initialised assistant ready to accept questions.

---

## `run_validation_test_cases(app: CrimeKGAI) -> None`

### Purpose
Demonstrate and verify that the Cypher validation layer catches three common categories of
bad queries — without running a full automated test framework.

### Test Cases

| Title | Question | Cypher | Expected behaviour |
|-------|----------|--------|--------------------|
| Reversed relationship direction | "Which locations are in area A1?" | `(Area)-[:LOCATION_IN_AREA]->(Location)` | Score < 0.7 because the relationship direction is reversed (it should be `(Location)-[:LOCATION_IN_AREA]->(Area)`). |
| Non-existent label | "Show me all suspects." | `MATCH (s:Suspect)` | Score < 0.7 because `:Suspect` is not a label in the schema. |
| Wrong relationship | "Show me connections between people." | `[:CONNECTED_TO]` | Score < 0.7 because `:CONNECTED_TO` does not exist in the schema. |

### How It Works
For each `(title, question, cypher)` tuple, calls `app._validate_cypher(question, cypher)` and
prints the score, issues, and corrected Cypher (if any). This is an interactive smoke test —
it confirms that the LLM-based validator behaves as expected with the live model.

### Inputs
`app` — an initialised `CrimeKGAI` instance.
### Output
None (prints to stdout).

---

## `run_demo(app: CrimeKGAI) -> None`

### Purpose
Exercise the assistant end-to-end with five representative questions, one for each retrieval
route, to confirm the full pipeline is working.

### Demo Questions and Expected Routes

| Question | Expected route |
|----------|---------------|
| "Who investigated crimes in area A1?" | GRAPH |
| "List crimes that occurred at a given location address." | GRAPH |
| "Recommend crimes similar to burglary." | VECTOR |
| "Find crimes similar to burglary but only those investigated by a specific officer." | HYBRID |
| "Which person has the most connections in this database and who are their top connections?" | AGENT |

### How It Works
Iterates over the questions list. For each question:
1. Prints `"Q: <question>"`.
2. Calls `app.answer(question)` inside a `try/except`.
3. Prints `"A: <answer>"` or `"A: error: <exception>"`.

### Inputs
`app` — an initialised `CrimeKGAI` instance.
### Output
None (prints to stdout).

---

## `run_repl(app: CrimeKGAI) -> None`

### Purpose
Provide an interactive command-line interface where a user can type questions and receive answers
in a continuous loop.

### How It Works
1. Prints a welcome message: `"Crime KG Assistant (type 'exit' to quit)."`.
2. Enters a `while True` loop:
   - Reads a line from stdin with `input("\n> ")`.
   - `EOFError` is caught to handle piped input gracefully (non-interactive use).
   - Skips empty lines.
   - Exits cleanly on `"exit"` or `"quit"`.
   - Calls `app.answer(question)` and prints the result.
   - Catches all exceptions and prints the error message so the REPL survives a single bad
     question without crashing.

### Inputs
`app` — an initialised `CrimeKGAI` instance.
### Output
None (streams responses to stdout).

### Why Catch `EOFError` Separately?
When the program is run with piped input (e.g. `echo "question" | python main.py`), Python's
`input()` raises `EOFError` when the pipe is exhausted. Catching it allows the REPL to terminate
gracefully instead of printing a traceback.
