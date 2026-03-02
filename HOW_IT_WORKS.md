# How It Works

This document is written for someone who has never seen this project before.
It explains **every module and utility** in the `crime_agent/` package, plus the two
top-level entry points, covering what each file does, why specific functions and
libraries were chosen, and the exact inputs and outputs of every significant piece
of code.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Entry Points](#2-entry-points)
   - [main.py](#mainpy)
   - [setup_embeddings.py](#setup_embeddingspy)
3. [crime_agent/utils.py](#3-crime_agentutilspy)
4. [crime_agent/prompts.py](#4-crime_agentpromptspy)
5. [crime_agent/embedding_setup.py](#5-crime_agentembedding_setuppy)
6. [crime_agent/app.py](#6-crime_agentapppy)
7. [crime_agent/assistant.py](#7-crime_agentassistantpy)
8. [Data-flow Diagram](#8-data-flow-diagram)

---

## 1. Project Overview

**DB-Assist** is a natural-language question-answering system for a crime
knowledge graph stored in **Neo4j**.  A user types a plain-English question
(e.g. *"Who investigated crimes in area A1?"*) and the system:

1. Decides **how** to answer it (graph traversal, vector similarity, a
   combination, or a multi-step AI agent).
2. Translates the question into a **Cypher query** (Neo4j's query language)
   and/or a **vector-similarity search**.
3. Validates and, if necessary, auto-corrects the query.
4. Runs the query against the Neo4j database.
5. Formats the raw database results into a readable English answer using
   Google's **Gemini** large-language model (LLM).

The LLM powering every AI step is **Gemini** (via `langchain_google_genai`).
Conversation context is kept alive across multiple questions using
**LangChain's `ConversationBufferMemory`**.

---

## 2. Entry Points

### `main.py`

**Purpose:** The command-line interface (CLI) that users run to interact with
the assistant.

```
python main.py [--demo] [--run-validation-tests] [--verbose]
```

#### How it works

1. `parse_args()` uses Python's built-in `argparse` to read three optional
   flags from the command line:

   | Flag | Effect |
   |------|--------|
   | `--demo` | Runs a fixed set of sample questions and prints the answers. |
   | `--run-validation-tests` | Runs three hard-coded Cypher validation scenarios and prints scores and corrections. |
   | `--verbose` | Makes the assistant print internal routing decisions, generated Cypher, and validation scores. |

2. `build_app(verbose=…)` (defined in `crime_agent/app.py`) is called to
   create and return the fully configured `CrimeKGAI` assistant object.

3. Depending on the flags one of three functions is called:
   - `run_validation_test_cases(app)` – prints validation output and exits.
   - `run_demo(app)` – iterates over the demo questions and exits.
   - `run_repl(app)` – starts an interactive read-eval-print loop so the user
     can type questions one by one until they type `exit`.

**Input:** Command-line arguments.  
**Output:** Printed text to `stdout`.

---

### `setup_embeddings.py`

**Purpose:** A standalone script that generates and stores vector embeddings
on Neo4j nodes so that similarity search can work.

```
python setup_embeddings.py [--index-name …] [--node-label …] [--text-props …] …
```

#### How it works

It is a thin wrapper that:

1. Calls `parse_args()` from `crime_agent/embedding_setup.py` to build an
   `EmbeddingSetupConfig` object from CLI arguments.
2. Calls `run_embedding_setup(config)` which does all the heavy lifting (see
   [Section 5](#5-crime_agentembedding_setuppy)).

**Input:** CLI arguments / environment variables.  
**Output:** Progress messages printed to `stdout`; embeddings and a vector
index written to Neo4j.

---

## 3. `crime_agent/utils.py`

**Purpose:** A collection of small, pure helper functions used throughout the
rest of the codebase. Keeping them here avoids repeating the same logic in
multiple files.

---

### `llm_content_to_text(content: Any) -> str`

**Why it exists:** The Gemini LLM sometimes returns its response as a plain
string, and sometimes as a list of content blocks (each being a dict with a
`"text"` key). This function hides that inconsistency so the rest of the code
can always work with a plain `str`.

**Input:** `content` – anything that the LLM might return as its `.content`
attribute.

| Type of `content` | Behaviour |
|---|---|
| `str` | Returned as-is. |
| `list` | Each element that is a `dict` with a `"text"` key contributes its text; other elements are converted with `str()`. All parts are joined with `"\n"`. |
| Anything else | Converted with `str()`. |

**Output:** A single `str` containing the LLM's text.

---

### `extract_json_object(text: str) -> Dict[str, Any]`

**Why it exists:** LLMs often surround their JSON output with Markdown
code-fences (`` ```json … ``` ``) and extra whitespace. This function strips
all of that and returns a Python dictionary.

**How it works step by step:**

1. Strip leading/trailing whitespace.
2. If the text starts with ` ``` `, use a regular expression to remove the
   opening fence (`` ```json\n ``) and the closing fence (`` \n``` ``).
3. Find the first `{` (`start`) and the last `}` (`end`) in the remaining
   text.  This deliberately ignores any text before the JSON object and after
   it (e.g. stray words).
4. If either delimiter is missing or `end ≤ start`, raise a `ValueError` with
   a preview of the bad text so the caller can debug it.
5. Call `json.loads()` on the slice `text[start : end+1]` and return the
   resulting `dict`.

**Input:** `text` – a string that should contain a JSON object, possibly
wrapped in Markdown.  
**Output:** `Dict[str, Any]` – the parsed JSON object.  
**Raises:** `ValueError` if no JSON object can be found; `json.JSONDecodeError`
if the extracted text is not valid JSON.

---

### `is_readonly_cypher(cypher: str) -> bool`

**Why it exists:** The system only ever wants to *read* from the database.
Allowing write operations (CREATE, DELETE, etc.) would be dangerous. This
function acts as a safety gate before any Cypher query is executed.

**How it works:**

1. Normalise the Cypher string: convert to upper-case and collapse all
   whitespace runs to a single space using `re.sub(r"\s+", " ", …)`. This
   prevents bypassing the check with unusual spacing.
2. Check whether any of the following tokens appears as a substring in the
   normalised string:
   `CREATE`, `MERGE`, `DELETE`, `DETACH`, `SET`, `DROP`, `ALTER`,
   `LOAD CSV`, `CALL dbms`, `CALL apoc`.
3. Return `True` (safe) if **none** of the tokens are found; `False`
   (unsafe) otherwise.

**Input:** `cypher` – a Cypher query string.  
**Output:** `bool` – `True` means read-only and safe to execute; `False`
means it contains a write or admin operation.

---

### `safe_index_name(name: str) -> str`

**Why it exists:** Neo4j vector index names are injected directly into Cypher
strings.  An attacker (or careless user) could provide a name containing
special characters that break the query or cause Cypher injection.  This
function validates the name before it is ever used.

**How it works:** Uses `re.fullmatch(r"[A-Za-z0-9_\-]+", name)` to check that
the name contains only letters, digits, underscores, and hyphens.  If the
pattern does not match the entire string, a `ValueError` is raised
immediately.  Otherwise the (already safe) name is returned unchanged.

**Input:** `name` – a proposed vector index name.  
**Output:** The same `name` string if it is valid.  
**Raises:** `ValueError` if the name contains any disallowed character.

---

### `serialize_value(value: Any) -> Any`

**Why it exists:** Neo4j returns Python objects (nodes, relationships,
`neo4j.time` types, etc.) that are not JSON-serialisable.  Before passing
query results to `json.dumps()` or to the LLM, everything must be converted
to plain Python primitives.

**How it works (recursive):**

| Type of `value` | Behaviour |
|---|---|
| `str`, `int`, `float`, `bool`, `None` | Returned as-is. |
| `list` | Each element is recursively serialised; a new list is returned. |
| `dict` | Each value is recursively serialised; a new dict is returned. |
| Object with `._properties` dict | That dict is recursively serialised (covers Neo4j `Node` and `Relationship` objects). |
| Object with `.items()` | Converted to `dict` first, then recursively serialised. |
| Anything else | Converted to `str`. |

**Input:** `value` – any Python value that may contain Neo4j types.  
**Output:** A JSON-safe Python value (nested `dict`/`list`/primitive).

---

## 4. `crime_agent/prompts.py`

**Purpose:** Defines all six **prompt templates** used to talk to the LLM.
Centralising them here keeps the business logic in `assistant.py` clean and
makes it easy to tweak wording without hunting through multiple files.

### `PromptSet` (dataclass)

A simple container that groups the six templates:

| Field | Role |
|---|---|
| `router` | Decides which search strategy to use. |
| `cypher_gen` | Generates a Cypher query from a natural-language question. |
| `cypher_validate` | Scores and optionally corrects a Cypher query. |
| `answer` | Turns raw database results into a human-readable answer. |
| `cypher_fix` | Fixes a Cypher query that threw a Neo4j error at runtime. |
| `hybrid_filter` | Writes a Cypher filter query for the hybrid search path. |

---

### `build_prompt_set() -> PromptSet`

**Why `ChatPromptTemplate`?** LangChain's `ChatPromptTemplate` lets the code
build a list of *chat messages* (system, human, AI) with placeholder
variables (e.g. `{schema}`, `{question}`) that are filled in at call time.
`MessagesPlaceholder("chat_history")` injects the entire conversation history
as a sequence of real message objects—preserving roles (human vs AI)—which
Gemini understands natively.

Each prompt is described below:

#### `router` prompt

- **System message:** Instructs the LLM to act as a router.  Explains the
  five possible routes (`GRAPH`, `VECTOR`, `HYBRID`, `AGENT`, `AMBIGUOUS`)
  and when to use each.
- **Human message template:** Provides the Neo4j `{schema}` and the
  `{question}`.
- **Expected LLM output:** A single JSON object:
  ```json
  { "route": "GRAPH", "reason": "…", "clarification_question": "" }
  ```

#### `cypher_gen` prompt

- **System message:** Constrains the LLM to produce read-only Cypher using
  only labels/properties that exist in the schema; always include `RETURN`;
  prefer `LIMIT 50`.
- **Human message template:** `{schema}`, `{question}`.
- **Expected LLM output:** A raw Cypher string (no Markdown).

#### `cypher_validate` prompt

- **System message:** Asks the LLM to score the Cypher (0–1), list issues,
  and provide a corrected version if fixable.
- **Human message template:** `{schema}`, `{question}`, `{cypher}`.
- **Expected LLM output:**
  ```json
  { "score": 0.85, "issues": [], "corrected_cypher": "" }
  ```

#### `answer` prompt

- **System message:** Tells the LLM it is a crime investigation assistant
  and that it should produce a human-friendly answer from raw JSON results.
- **Human message template:** `{question}`, `{results}`.
- **Expected LLM output:** A plain-English paragraph.

#### `cypher_fix` prompt

- **System message:** Like `cypher_gen` but specialised for fixing a failed
  query.
- **Human message template:** `{schema}`, `{question}`, `{cypher}` (the
  broken query), `{error}` (the Neo4j error message).
- **Expected LLM output:** A corrected Cypher string.

#### `hybrid_filter` prompt

- **System message:** Instructs the LLM to write a Cypher query that starts
  with `WITH $candidate_ids AS candidate_ids` and filters the candidate nodes
  from a prior vector search using graph relationships.
- **Human message template:** `{schema}`, `{question}`.
- **Expected LLM output:** A Cypher string that references `$candidate_ids`.

---

## 5. `crime_agent/embedding_setup.py`

**Purpose:** Generates **vector embeddings** for Neo4j nodes (by default
`Crime` nodes) using the Gemini embedding model and stores them back in the
database, then creates a Neo4j vector index so that similarity queries can
run efficiently.

This is a **one-time setup** step; it does not need to run on every
application start, only when new nodes are added or when the index does not
yet exist.

---

### `EmbeddingSetupConfig` (dataclass)

Holds all configuration for one embedding run:

| Field | Default | Meaning |
|---|---|---|
| `index_name` | `"crime_embeddings"` | Name of the Neo4j vector index to create. |
| `node_label` | `"Crime"` | Which node label to embed. |
| `text_props` | `["type", "last_outcome"]` | Node properties whose text is concatenated to form the embedding input. |
| `embedding_prop` | `"embedding"` | Property name on each node where the vector will be stored. |
| `limit` | `2000` | Maximum number of nodes to process in one run. |
| `batch_size` | `64` | How many texts to send to the embedding API at once. |

---

### `parse_args() -> EmbeddingSetupConfig`

Reads CLI flags (falling back to environment variables, then hard-coded
defaults) and returns an `EmbeddingSetupConfig`.  Raises `SystemExit` if
`--text-props` resolves to an empty list.

**Input:** `sys.argv` (CLI arguments) and environment variables.  
**Output:** `EmbeddingSetupConfig`.

---

### `ensure_required_env() -> None`

Checks that `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and
`GEMINI_API_KEY` are all present in the process environment.  Raises
`SystemExit` with a clear message listing any missing variables.

---

### `build_driver() -> neo4j.Driver`

Creates and returns a raw **Neo4j `Driver`** (from the official `neo4j`
Python package) using the connection details from environment variables.

**Why a raw driver instead of LangChain's `Neo4jGraph`?**  
`Neo4jGraph` is a higher-level LangChain wrapper for reading data and schema.
For *writing* embedding vectors back to nodes we need the lower-level driver
which supports explicit sessions and write transactions.

**Output:** `neo4j.Driver` (connection must be closed by the caller).

---

### `build_embeddings() -> GoogleGenerativeAIEmbeddings`

Initialises the **Gemini embedding model** (`gemini-embedding-001`) via
LangChain's `langchain_google_genai` wrapper and prints a confirmation
message.

**Output:** `GoogleGenerativeAIEmbeddings` – an object with
`.embed_query(text)` and `.embed_documents(texts)` methods.

---

### `ensure_vector_index(session, config) -> None`

Runs a single Cypher statement:

```cypher
CREATE VECTOR INDEX <index_name> IF NOT EXISTS
FOR (n:<node_label>) ON (n.<embedding_prop>)
OPTIONS { indexConfig: { `vector.dimensions`: 768,
                         `vector.similarity_function`: 'cosine' } }
```

**Why 768 dimensions?** The `gemini-embedding-001` model produces 768-
dimensional vectors.

**Why cosine similarity?** Cosine similarity measures the angle between two
vectors; it is scale-invariant and works well for text embeddings where the
magnitude of the vector is not meaningful.

**Why `IF NOT EXISTS`?**  Makes the operation idempotent—safe to call
multiple times.

**Input:** An open Neo4j `Session` and an `EmbeddingSetupConfig`.  
**Output:** None (side-effect: index created in Neo4j if absent).

---

### `fetch_nodes_to_embed(session, config) -> List[Tuple[str, str]]`

Queries Neo4j for nodes of the configured label that do **not** yet have an
embedding stored:

```cypher
MATCH (n:<node_label>)
WHERE n.<embedding_prop> IS NULL
RETURN elementId(n) AS eid, properties(n) AS props
LIMIT <limit>
```

For each returned node it builds an **embedding input string** by joining the
configured text properties:

```
type: Burglary | last_outcome: Under investigation
```

If none of the configured properties exist on a node, it falls back to
`str(props)` so no node is silently skipped.

**Input:** An open Neo4j `Session` and an `EmbeddingSetupConfig`.  
**Output:** `List[Tuple[str, str]]` – each tuple is `(element_id, text_to_embed)`.

---

### `update_embeddings(session, embeddings, config, nodes) -> int`

Processes `nodes` in batches of `config.batch_size`:

1. Calls `embeddings.embed_documents(texts)` to get a list of vectors from
   the Gemini API in one API call per batch (reduces latency vs. one call per
   node).
2. Writes the vectors back to Neo4j with a single parameterised Cypher
   statement per batch using `UNWIND`:

   ```cypher
   UNWIND $rows AS row
   MATCH (n) WHERE elementId(n) = row.eid
   SET n.<embedding_prop> = row.embedding
   ```

3. Prints progress after each batch.

**Why batch processing?** Embedding APIs charge per token and have rate
limits; batching is both faster and more economical.

**Input:** Open Neo4j `Session`, `GoogleGenerativeAIEmbeddings`, `EmbeddingSetupConfig`, and the node list from `fetch_nodes_to_embed`.  
**Output:** `int` – the total number of nodes updated.  
**Side-effect:** Each processed node now has an `embedding` property in Neo4j.

---

### `run_embedding_setup(config: EmbeddingSetupConfig) -> None`

Top-level orchestrator:

1. Loads `.env` into the environment (`load_dotenv`).
2. Validates required environment variables (`ensure_required_env`).
3. Initialises the embedding model and Neo4j driver.
4. Opens a driver session and calls `ensure_vector_index`, then
   `fetch_nodes_to_embed`.  If no nodes need embedding it exits early.
5. Calls `update_embeddings` and prints a final summary.
6. Always closes the Neo4j driver (via `try/finally`).

**Input:** `EmbeddingSetupConfig`.  
**Output:** None (prints progress; side-effects in Neo4j).

---

## 6. `crime_agent/app.py`

**Purpose:** Wires all external dependencies (Neo4j, Gemini LLM, Gemini
embeddings, memory) together into a `CrimeKGAI` instance, and provides the
three runtime modes (REPL, demo, validation).

---

### `build_app(verbose: bool) -> CrimeKGAI`

1. **`load_dotenv()`** – Reads a `.env` file in the working directory so that
   credentials do not have to be set as shell environment variables.
2. **Environment validation** – Checks for `NEO4J_URI`, `NEO4J_USERNAME`,
   `NEO4J_PASSWORD`, `GEMINI_API_KEY`; raises `SystemExit` if any are absent.
3. **`Neo4jGraph`** – LangChain's high-level wrapper that auto-introspects the
   schema (node labels, relationship types, property keys) and exposes a
   `.query()` method.  The schema is later injected into every LLM prompt.
4. **`ChatGoogleGenerativeAI`** – The Gemini chat model.  `temperature=0`
   makes responses deterministic and consistent, which is important for
   structured JSON outputs (router, validator).
5. **`GoogleGenerativeAIEmbeddings`** – The Gemini embedding model used for
   vector similarity searches at query time.
6. **`ConversationBufferMemory`** – Stores the full message history so that
   follow-up questions can reference earlier context.  `return_messages=True`
   keeps messages as typed objects (not a string summary) for accurate
   multi-turn conversations.
7. **`index_name`** – Reads the vector index name from `NEO4J_VECTOR_INDEX`
   (default: `"crime_embeddings"`).

**Input:** `verbose: bool`.  
**Output:** A fully initialised `CrimeKGAI` object.

---

### `run_validation_test_cases(app: CrimeKGAI) -> None`

Runs three deliberately incorrect Cypher queries through the validator to
demonstrate the validation and auto-correction pipeline:

| Test case | Fault |
|---|---|
| Reversed relationship direction | Correct direction is `(l)-[:LOCATION_IN_AREA]->(a)` but query has it reversed. |
| Non-existent label | `Suspect` does not exist in the schema. |
| Ambiguous relationship | `CONNECTED_TO` does not exist. |

For each case it prints the validation `score`, `issues`, and any
`corrected_cypher` suggested by the LLM.

**Input:** `CrimeKGAI`.  **Output:** Printed text to `stdout`.

---

### `run_demo(app: CrimeKGAI) -> None`

Iterates over five representative questions covering all four routing paths
(graph, vector, hybrid, agent) and prints each answer (or the error if one
occurs).

---

### `run_repl(app: CrimeKGAI) -> None`

Reads questions from `stdin` in a loop.  Stops on `EOF` (e.g. `Ctrl-D`) or
when the user types `exit`/`quit`.  Each answer (or error message) is printed
immediately.

---

## 7. `crime_agent/assistant.py`

**Purpose:** The core intelligence of the system.  The `CrimeKGAI` class
orchestrates routing, Cypher generation, validation, execution, vector search,
hybrid search, agent-mode reasoning, and answer formatting.

---

### `ValidationResult` (dataclass)

```python
@dataclass
class ValidationResult:
    score: float          # 0.0–1.0; ≥0.7 is considered executable
    issues: List[str]     # human-readable descriptions of problems
    corrected_cypher: Optional[str]  # LLM-suggested fix, or None
```

---

### `CrimeKGAI.__init__`

Stores all injected dependencies and calls `build_prompt_set()` once to
create the six prompt templates.  All prompts are attributes of the instance
(`self._router_prompt`, etc.) for easy access.

---

### `_chat_history() -> List[BaseMessage]`

Loads the current conversation from `ConversationBufferMemory` and returns it
as a list of LangChain `BaseMessage` objects.  Falls back gracefully to an
empty list if the memory is empty or returns a plain string instead of a list.

**Why?** `MessagesPlaceholder` in the prompts expects typed message objects.

---

### `classify(question: str) -> Dict[str, Any]`

**Purpose:** Decides *how* to answer the question.

1. Fills `_router_prompt` with the Neo4j schema, the question, and chat
   history.
2. Invokes the LLM.
3. Parses the JSON response with `extract_json_object`.
4. Validates the `route` value against the allowed set
   `{"GRAPH", "VECTOR", "HYBRID", "AGENT", "AMBIGUOUS"}`; defaults to
   `"GRAPH"` if invalid.

**Input:** `question: str`.  
**Output:** `{"route": str, "reason": str, "clarification_question": str}`.

---

### `_generate_cypher(question: str) -> str`

**Purpose:** Translates a natural-language question into a Cypher query.

Fills `_cypher_gen_prompt`, invokes the LLM, and returns the raw text
(stripped of whitespace).  No validation happens here—that is the next step.

**Input:** `question: str`.  
**Output:** A Cypher query string (may contain errors).

---

### `_validate_cypher(question: str, cypher: str) -> ValidationResult`

**Purpose:** Checks whether a Cypher query is correct and safe before running
it.

**Step 1 – Safety check:** `is_readonly_cypher(cypher)`.  If the query is not
read-only, immediately returns `ValidationResult(score=0.0, issues=["Query is
not read-only"])` without calling the LLM.

**Step 2 – LLM validation:** Fills `_cypher_validate_prompt` and parses the
JSON response.

**Step 3 – Critical-issue cap:** Regardless of what score the LLM gives, if
any of the following words appear in the issues list—`"does not exist"`,
`"not exist"`, `"reversed"`, `"wrong direction"`, `"missing return"`,
`"no return"`, `"label"`, `"relationship type"`, `"property"`—the score is
capped at `0.69`.  This prevents a query with a fundamental schema error from
being executed just because the LLM gave it a generous score.

These specific phrases were chosen because they each signal a *fundamental*
structural problem rather than a stylistic one:

- `"does not exist"` / `"not exist"` – a label, relationship type, or
  property named in the query is not in the schema; the query will always
  return zero rows or throw a runtime error.
- `"reversed"` / `"wrong direction"` – Neo4j relationship directions are
  strict; a reversed arrow means the `MATCH` finds nothing.
- `"missing return"` / `"no return"` – Cypher requires an explicit `RETURN`
  clause; without it the query is syntactically invalid.
- `"label"` / `"relationship type"` / `"property"` – broad catch-all terms
  the validator uses when it identifies that a key part of the query schema is
  wrong.

Capping at `0.69` (just below the `0.7` execution threshold) forces the code
into the corrective path rather than attempting execution, while still leaving
room for the `corrected_cypher` path (`0.4 ≤ score < 0.7`) to be tried.

**Input:** `question: str`, `cypher: str`.  
**Output:** `ValidationResult`.

---

### `_execute_cypher_with_validation(question, cypher, params) -> Tuple[str, Any]`

**Purpose:** Execute a Cypher query, automatically trying to fix or correct it
if confidence is low.

```
score ≥ 0.7  →  execute; if Neo4j throws an error, ask LLM to fix → validate → execute
0.4 ≤ score < 0.7  →  use corrected_cypher from validator → validate again → execute
score < 0.4  →  raise ValueError asking user to clarify
```

**Why two repair paths?** LLM-generated code frequently has small mistakes
(wrong case, reversed relationship).  The first path handles *runtime* errors
(syntax the LLM didn't realise was wrong); the second path handles *schema*
errors the LLM already noticed and corrected during validation.

**Input:** `question: str`, `cypher: str`, `params: Optional[Dict]`.  
**Output:** `Tuple[str, Any]` – `(executed_cypher, rows_from_neo4j)`.  
**Raises:** `ValueError` if no valid query can be produced.

---

### `_format_answer(question: str, results: Any) -> str`

Serialises Neo4j results with `serialize_value`, encodes them as JSON, then
asks the LLM (via `_answer_prompt`) to write a plain-English answer.

**Input:** `question: str`, `results: Any` (raw Neo4j rows).  
**Output:** Human-readable answer string.

---

### `vector_search(question, k=8, min_score=0.7) -> List[Dict]`

**Purpose:** Find nodes whose stored embeddings are semantically close to the
question.

1. `self.embeddings.embed_query(question)` – converts the question into a
   768-dimensional vector.
2. Runs the Neo4j built-in procedure `db.index.vector.queryNodes` to retrieve
   the top-`k` nodes by cosine similarity, filtered to `score ≥ min_score`.
3. Returns a list of dicts: `{id, labels, properties, score}`.

**Why k=8?** Eight candidates typically cover the relevant results without
overwhelming the LLM with noise.

**Why min_score=0.7?** Cosine similarity of 0.7 filters out weakly related
nodes while still returning semantically meaningful matches.

**Input:** `question: str`, `k: int`, `min_score: float`.  
**Output:** `List[Dict[str, Any]]`.

---

### `hybrid_search(question, k=12, min_score=0.7) -> Any`

**Purpose:** Combines vector similarity with graph constraints.

1. `vector_search(question, k=k, min_score=min_score)` – retrieves candidate
   node IDs.
2. If no candidates are found, returns `[]`.
3. Asks the LLM (via `_hybrid_filter_prompt`) to write a Cypher query that
   starts from those candidate IDs and applies additional graph-based filters.
4. Validates the generated Cypher:
   - If it is empty, not read-only, or doesn't reference `$candidate_ids`,
     falls back to returning the raw vector candidates.
5. Calls `_execute_cypher_with_validation` with `params={"candidate_ids": …}`.

**Why k=12 here vs k=8 for pure vector?** The graph filter step may discard
some candidates, so we start with a larger pool.

**Input:** `question: str`, `k: int`, `min_score: float`.  
**Output:** Neo4j rows or the raw candidate list.

---

### `answer(question: str) -> str`

**Purpose:** The main public method—handles a single user question end-to-end.

**Special case:** If the question matches a known ambiguous phrase
(`"show me connections between people"`) it immediately returns a
clarification message without calling the LLM router.  This avoids generating
bad Cypher for a question that can never be answered without more context.

**Routing:**

| Route | Action |
|---|---|
| `AMBIGUOUS` | Return the LLM's clarification question; save to memory. |
| `VECTOR` | `vector_search` → `_format_answer` → save to memory. |
| `HYBRID` | `hybrid_search` → `_format_answer` → save to memory. |
| `AGENT` | `_agent_answer` → save to memory. |
| `GRAPH` (default) | `_generate_cypher` → `_execute_cypher_with_validation` → `_format_answer` → save to memory. |

Every path ends with `self.memory.save_context(…)` to keep the conversation
history up to date.

**Input:** `question: str`.  
**Output:** Answer string.

---

### `_agent_answer(question: str) -> str`

**Purpose:** Used when the question requires multi-step reasoning—e.g.
"find the person with the most connections *and* list their top connections",
which requires knowing the first answer before forming the second query.

**How it works:**

1. Defines three **LangChain tools** as inner functions (using the `@tool`
   decorator):
   - `cypher_executor(cypher)` – validates and runs arbitrary Cypher; returns
     JSON rows.
   - `vector_search(query)` – runs vector similarity search; returns JSON.
   - `hybrid_search(query)` – runs hybrid search; returns JSON.

2. `create_react_agent(self.llm, tools=[…])` builds a **ReAct agent**—an
   LLM that can decide which tool to call, call it, inspect the result, and
   decide whether to call another tool or produce a final answer.

3. The agent is invoked with the full conversation history plus the new
   question.

4. The final state's `messages` list is scanned in reverse for the last
   `AIMessage`.  That message's content is returned as the answer.

**Why ReAct?** Complex multi-step questions cannot be solved by a single
Cypher query.  ReAct (Reason + Act) agents iteratively plan, act, and observe
until they have enough information, which is the appropriate strategy for
open-ended questions.

**Failure / termination behaviour:**

- **No final AI message found:** If `messages` contains no `AIMessage` at all
  (e.g. the agent hit an internal error before producing output), the method
  returns the fallback string `"I couldn't complete that reasoning step.
  Please rephrase the question."` rather than raising an exception.
- **Maximum iterations:** LangGraph's `create_react_agent` enforces a default
  recursion/step limit (typically 25).  If the agent has not finished within
  that limit it raises a `GraphRecursionError`.  This surfaces as an exception
  from `answer()`, which `run_repl` catches and prints as an error message.
- **Tool errors:** Each tool (`cypher_executor`, `vector_search`,
  `hybrid_search`) has its own `try/except` block that returns a JSON error
  string instead of raising.  This allows the agent to see the error, reason
  about it, and potentially try a different query rather than crashing.

**Input:** `question: str`.  
**Output:** Answer string.

---

## 8. Data-flow Diagram

```
User types question
        │
        ▼
  answer(question)
        │
        ├──[AMBIGUOUS]──► return clarification_question
        │
        ├──[VECTOR]──► embed question ──► db.index.vector.queryNodes
        │                                          │
        │                                  _format_answer ──► return
        │
        ├──[HYBRID]──► vector_search ──► hybrid_filter Cypher
        │                                          │
        │                               _execute_cypher_with_validation
        │                                          │
        │                                  _format_answer ──► return
        │
        ├──[AGENT]──► create_react_agent (ReAct loop)
        │               ├── cypher_executor tool
        │               ├── vector_search tool
        │               └── hybrid_search tool
        │                          │
        │                  last AIMessage ──► return
        │
        └──[GRAPH]──► _generate_cypher
                              │
                  _execute_cypher_with_validation
                    ├── score ≥ 0.7 ──► graph.query()
                    │       └── error ──► LLM fix ──► re-validate ──► graph.query()
                    ├── 0.4–0.69 ──► use corrected_cypher ──► re-validate ──► graph.query()
                    └── < 0.4 ──► raise ValueError
                              │
                      _format_answer ──► return
```
