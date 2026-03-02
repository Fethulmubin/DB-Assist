# HOW IT WORKS — `assistant.py`

This is the core of the project. It defines the `CrimeKGAI` class, which wires together the
Neo4j graph database, the Gemini LLM, vector embeddings, conversation memory, and all the prompt
templates into a single object that can answer natural-language questions about crime data.

---

## Dependencies

| Import | Why it is used |
|--------|----------------|
| `ConversationBufferMemory` | Stores the full conversation history as a list of `HumanMessage` / `AIMessage` objects so every call to the LLM can include prior turns. |
| `Neo4jGraph` | LangChain wrapper around the Neo4j Python driver. Exposes `.schema` (auto-fetched) and `.query(cypher, params)`. |
| `ChatGoogleGenerativeAI` | The Gemini chat model. Called with `.invoke(messages)` to get a response. |
| `GoogleGenerativeAIEmbeddings` | Converts text to a 768-dimensional float vector for semantic search. |
| `create_react_agent` | LangGraph utility that builds a ReAct (Reason + Act) loop agent from an LLM and a list of tools. |
| `build_prompt_set` | Returns the six `ChatPromptTemplate` objects defined in `prompts.py`. |
| `utils.*` | The five helper functions from `utils.py`. |

---

## `ValidationResult` Dataclass

```python
@dataclass
class ValidationResult:
    score: float
    issues: List[str]
    corrected_cypher: Optional[str] = None
```

A plain container for the three fields the LLM returns from the `cypher_validate` prompt. Using
a dataclass instead of a raw `dict` means the rest of the code can access `.score`, `.issues`,
and `.corrected_cypher` with IDE completion and avoids `KeyError` mistakes.

---

## `CrimeKGAI` Class

### Constructor `__init__`

```python
def __init__(self, graph, llm, embeddings, vector_index_name, memory, verbose=False)
```

**Parameters:**

| Parameter           | Type                         | Description |
|---------------------|------------------------------|-------------|
| `graph`             | `Neo4jGraph`                 | Connected Neo4j session with auto-loaded schema. |
| `llm`               | `ChatGoogleGenerativeAI`     | The Gemini model instance. |
| `embeddings`        | `GoogleGenerativeAIEmbeddings` | The embedding model instance. |
| `vector_index_name` | `str`                        | Name of the vector index in Neo4j. Validated with `safe_index_name` immediately. |
| `memory`            | `ConversationBufferMemory`   | Conversation history store. |
| `verbose`           | `bool`                       | If `True`, prints routing decisions, Cypher, and validation scores to stdout. |

The constructor also calls `build_prompt_set()` and assigns the six prompts to instance
attributes, so the prompt objects are created once and reused across all method calls.

---

### `_chat_history() -> List[BaseMessage]`

**Purpose:** Retrieve the conversation history from `memory` in the format that LangChain prompt
templates expect (a list of `BaseMessage` objects).

**How it works:**
1. Calls `self.memory.load_memory_variables({})` which returns a dict usually containing
   a `"history"` key.
2. If the value is already a `list` (the normal case with `return_messages=True`), return it.
3. If it is a plain `str` (can happen when `return_messages=False`), wrap it in a
   `HumanMessage` list.
4. Otherwise return an empty list.

**Why needed:** `ConversationBufferMemory` can be configured in two modes. This helper insulates
the rest of the code from that difference.

---

### `classify(question: str) -> Dict[str, Any]`

**Purpose:** Ask the LLM to decide which retrieval strategy should handle this question —
`GRAPH`, `VECTOR`, `HYBRID`, `AGENT`, or `AMBIGUOUS`.

**How it works:**
1. Formats the `router` prompt with `schema`, `question`, and `chat_history`.
2. Sends the messages to the LLM with `llm.invoke(messages)`.
3. Converts the response to a string with `llm_content_to_text`.
4. Parses the JSON object with `extract_json_object`.
5. Validates that `route` is one of the five known values; defaults to `"GRAPH"` if not.
6. Returns a dict with `route`, `reason`, and `clarification_question`.

**Inputs:** `question` — the user's raw question string.
**Output:** `{"route": str, "reason": str, "clarification_question": str}`

---

### `_generate_cypher(question: str) -> str`

**Purpose:** Ask the LLM to translate the natural-language question into a Cypher `MATCH … RETURN`
query.

**How it works:**
1. Formats the `cypher_gen` prompt with `schema`, `question`, and `chat_history`.
2. Invokes the LLM.
3. Returns the response as a stripped plain string — the raw Cypher text.

**Inputs:** `question` — the user's question.
**Output:** `str` — a Cypher query (may still have errors; validation comes next).

---

### `_validate_cypher(question: str, cypher: str) -> ValidationResult`

**Purpose:** Score a Cypher query for correctness and get a corrected version if the score is
in the fixable range.

**How it works:**
1. First guard: calls `is_readonly_cypher(cypher)`. If the query contains write keywords, returns
   `ValidationResult(score=0.0, issues=["Query is not read-only"])` immediately without calling
   the LLM.
2. Formats the `cypher_validate` prompt and invokes the LLM.
3. Parses the JSON response to extract `score`, `issues`, and `corrected_cypher`.
4. If `corrected_cypher` itself fails `is_readonly_cypher`, it is set to `None` — the assistant
   will never execute a corrected query that contains write operations.
5. **Critical-marker downgrade:** even if the LLM gives a high score, if the issues list contains
   phrases like `"does not exist"`, `"wrong direction"`, `"missing return"` etc., the score is
   capped at `0.69` to force the correction branch. This protects against a model that rates its
   own flawed correction too generously.

**Inputs:**
| Parameter  | Description |
|------------|-------------|
| `question` | Original user question (gives the validator intent context). |
| `cypher`   | The Cypher string to check. |

**Output:** `ValidationResult` with `.score` (0–1), `.issues` (list of strings), and optionally
`.corrected_cypher`.

---

### `_execute_cypher_with_validation(question, cypher, params=None) -> Tuple[str, Any]`

**Purpose:** Execute a Cypher query, applying the validation score to decide whether to run it,
fix it, or reject it.

**Decision tree:**

```
validate(cypher)
 ├── score >= 0.7  → try graph.query(cypher)
 │    ├── success  → return (cypher, rows)
 │    └── Neo4j error → ask LLM to fix → re-validate
 │         ├── fixed score >= 0.7  → return (fixed_cypher, rows)
 │         └── else               → re-raise original exception
 │
 ├── 0.4 <= score < 0.7 AND corrected_cypher exists
 │    └── re-validate(corrected_cypher)
 │         ├── score >= 0.7  → return (corrected_cypher, rows)
 │         └── else          → raise ValueError (low confidence)
 │
 └── score < 0.4 (or no correction)
      └── raise ValueError("Low-confidence Cypher. Please clarify…")
```

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `question` | Original question (used in fix prompt and error messages). |
| `cypher`   | The Cypher string to attempt. |
| `params`   | Optional dict of Cypher parameters (e.g. `{"candidate_ids": […]}`). |

**Output:** `(executed_cypher: str, rows: Any)` — the actual query that ran and its results.

**Why this multi-step approach?**
A single pass often fails because: (a) the LLM generates a query with a minor schema mistake,
or (b) Neo4j raises a syntax error for a query the LLM rated as valid. The three-tier approach
(run → fix on error → correct on low score) recovers from both failure modes automatically.

---

### `_format_answer(question: str, results: Any) -> str`

**Purpose:** Ask the LLM to convert raw query results (Neo4j rows or vector search hits) into a
clear, human-readable answer.

**How it works:**
1. Calls `serialize_value(results)` to convert any Neo4j-specific objects to plain Python.
2. Calls `json.dumps(serialized, ensure_ascii=False)` to turn the data into a JSON string.
3. Formats the `answer` prompt with `question`, `results` (the JSON string), and `chat_history`.
4. Invokes the LLM and returns the response as a stripped string.

**Inputs:**
| Parameter | Description |
|-----------|-------------|
| `question` | The original user question. |
| `results`  | Raw Neo4j query result rows or a list of vector search dicts. |

**Output:** `str` — a conversational answer ready to show to the user.

---

### `vector_search(question: str, k: int = 8, min_score: float = 0.7) -> List[Dict]`

**Purpose:** Find the `k` Neo4j nodes whose stored embedding is most similar to the question,
filtered by a minimum cosine similarity score.

**How it works:**
1. Calls `self.embeddings.embed_query(question)` to convert the question into a 768-dimensional
   float vector.
2. Runs the Neo4j built-in procedure `db.index.vector.queryNodes` with the question vector.
   The procedure returns `(node, score)` pairs ordered by cosine similarity descending.
3. The `WHERE score >= $min_score` clause filters out weak matches server-side.
4. Returns a list of dicts with keys `id`, `labels`, `properties`, and `score`.

**Inputs:**
| Parameter   | Default | Description |
|-------------|---------|-------------|
| `question`  | —       | The search query text. |
| `k`         | `8`     | Maximum number of neighbours to return from the index. |
| `min_score` | `0.7`   | Cosine similarity threshold (0–1). Only nodes above this score are returned. |

**Output:** `List[Dict]` — matched nodes with their labels, properties, and similarity score.

**Why use `db.index.vector.queryNodes` instead of manual cosine math?**
Neo4j's native vector index is implemented in the database engine (using HNSW under the hood),
so it scales to millions of nodes without loading all vectors into Python memory.

---

### `hybrid_search(question: str, k: int = 12, min_score: float = 0.7) -> Any`

**Purpose:** Combine vector similarity (to find semantically related nodes) with graph traversal
(to apply structural constraints from the question).

**How it works:**
1. Calls `vector_search(question, k, min_score)` to get candidate nodes.
2. Extracts the Neo4j `elementId` of each candidate.
3. If there are no candidates, returns `[]` immediately.
4. Uses the `hybrid_filter` prompt to ask the LLM to write a Cypher query that:
   - Receives `$candidate_ids` as a parameter.
   - Matches those candidates in the graph.
   - Adds additional constraints derived from the question.
5. Before executing, applies three safety checks on the LLM-generated Cypher:
   - Is it non-empty?
   - Is it read-only (`is_readonly_cypher`)?
   - Does it reference `$candidate_ids` or `candidate_ids` (ensuring the LLM actually uses the
     pre-filtered set)?
   If any check fails, falls back to returning the raw vector candidates.
6. Calls `_execute_cypher_with_validation` with the candidate IDs as parameters.

**Inputs:**
| Parameter   | Default | Description |
|-------------|---------|-------------|
| `question`  | —       | The search query. |
| `k`         | `12`    | Wider initial candidate pool (more than pure vector search) to give the graph filter something to work with. |
| `min_score` | `0.7`   | Minimum vector similarity for the first stage. |

**Output:** `Any` — either filtered graph rows or the raw vector candidates as a fallback.

---

### `answer(question: str) -> str`

**Purpose:** The main public entry point. Routes the question to the correct retrieval strategy,
executes the query, formats the answer, and saves the exchange to memory.

**How it works:**

```
1. Hard-coded check for "connections between people" → return clarification immediately.
2. classify(question) → get route.
3. AMBIGUOUS  → return clarification_question from classifier.
4. VECTOR     → vector_search → _format_answer.
5. HYBRID     → hybrid_search → _format_answer.
6. AGENT      → _agent_answer.
7. GRAPH (default) → _generate_cypher → _execute_cypher_with_validation → _format_answer.
8. memory.save_context(input, output) to record the exchange.
9. Return the answer string.
```

**Inputs:** `question` — the user's raw input string.
**Output:** `str` — the assistant's response (always a string, even for error cases).

**Why the hard-coded check for "connections between people"?**
This exact phrasing reliably produces an ambiguous, overly broad MATCH in the graph. Instead of
letting it reach the LLM and potentially return thousands of rows, the assistant proactively asks
the user to clarify which relationship type and which person they mean.

---

### `_agent_answer(question: str) -> str`

**Purpose:** Handle complex multi-step questions by giving the Gemini LLM three tools and letting
it decide autonomously how many queries to run and in what order.

**How it works:**
1. Defines three `@tool`-decorated inner functions:
   - **`cypher_executor(cypher)`** — validates and executes a single Cypher query, returns JSON rows.
   - **`vector_search(query)`** — calls `self.vector_search` and returns JSON.
   - **`hybrid_search(query)`** — calls `self.hybrid_search` and returns JSON.
2. Builds a ReAct agent with `create_react_agent(self.llm, tools=[…])`.
3. Constructs the initial state as `{"messages": [*history, HumanMessage(content=question)]}`.
4. Invokes the agent with `agent.invoke(state)`. The agent iterates:
   - **Reason** — the LLM decides which tool to call next.
   - **Act** — the tool is called and its output is appended to `messages`.
   - Repeat until the LLM decides it has enough information to answer.
5. Finds the last `AIMessage` in the final state's `messages` list and returns its content.

**Inputs:** `question` — the user's complex, multi-step question.
**Output:** `str` — the agent's final synthesised answer.

**Why use LangGraph `create_react_agent` instead of a hand-written loop?**
ReAct reliably handles dynamic reasoning chains. The number of tool calls needed for a complex
question like "Which person has the most connections and who are their top connections?" is not
known in advance; the agent figures it out on its own without the code needing to predict how
many steps are required.
