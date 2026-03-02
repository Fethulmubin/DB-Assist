# HOW IT WORKS — `prompts.py`

This file defines every prompt template used by the assistant. Templates are created with
LangChain's `ChatPromptTemplate` and stored in an immutable `PromptSet` dataclass so that the
rest of the code can retrieve any prompt by name without hard-coding strings in multiple places.

---

## The `PromptSet` Dataclass

```python
@dataclass
class PromptSet:
    router: ChatPromptTemplate
    cypher_gen: ChatPromptTemplate
    cypher_validate: ChatPromptTemplate
    answer: ChatPromptTemplate
    cypher_fix: ChatPromptTemplate
    hybrid_filter: ChatPromptTemplate
```

`PromptSet` is a simple container that groups the six prompts together. Using a dataclass instead
of a plain `dict` gives IDE auto-completion and makes it clear exactly which prompts exist.

---

## `build_prompt_set() -> PromptSet`

This is the single public function. It constructs all six `ChatPromptTemplate` objects and
returns them wrapped in a `PromptSet`. It takes no arguments and has no side-effects.

### Output
`PromptSet` — an instance containing all six ready-to-use prompt templates.

---

## The Six Prompts

### 1. `router` — Question Classifier

**Purpose:** Decide which retrieval strategy should answer the user's question.

**System message (condensed):**
> You are routing user questions for a Neo4j knowledge-graph assistant.
> Output a single JSON object with keys `route`, `reason`, and `clarification_question`.

**Route values and when each is chosen:**

| Route       | When to use |
|-------------|-------------|
| `GRAPH`     | The question requires explicit relationship traversal (e.g. "Who investigated crimes in area A1?"). |
| `VECTOR`    | The question asks for semantic similarity or recommendations (e.g. "Find crimes similar to burglary"). |
| `HYBRID`    | The question combines similarity **and** graph constraints (e.g. "Similar crimes but only in area A2"). |
| `AGENT`     | Complex multi-step reasoning where the number of queries depends on intermediate results. |
| `AMBIGUOUS` | The question cannot be executed safely — a clarifying follow-up is needed. |

**Human message variables:**
| Variable        | Filled by                          |
|-----------------|------------------------------------|
| `{schema}`      | `Neo4jGraph.schema` (auto-fetched) |
| `{question}`    | The user's raw question string     |
| `{chat_history}` | Previous turns via `MessagesPlaceholder` |

**Expected LLM output:**
```json
{
  "route": "GRAPH",
  "reason": "The question asks for a direct graph lookup.",
  "clarification_question": ""
}
```

**Why `MessagesPlaceholder("chat_history")`?**
Injecting the conversation history lets the router understand follow-up questions (e.g. "And what
about area B2?" after an earlier area A1 question).

---

### 2. `cypher_gen` — Cypher Query Generator

**Purpose:** Translate a natural-language question into a valid, read-only Cypher query.

**System message constraints (summarised):**
- Use only labels/types/properties that exist in the provided schema.
- Never write data (`CREATE`, `MERGE`, `DELETE`, `SET`, `DROP`).
- Always include an explicit `RETURN`.
- Default to `LIMIT 50` for potentially large results.
- Output only the Cypher, no markdown fencing.

**Human message variables:**
| Variable        | Filled by |
|-----------------|-----------|
| `{schema}`      | `Neo4jGraph.schema` |
| `{question}`    | The user's question |
| `{chat_history}` | Previous turns |

**Expected LLM output:**
```cypher
MATCH (o:Officer)-[:INVESTIGATED]->(c:Crime)-[:OCCURRED_AT]->(l:Location)-[:LOCATION_IN_AREA]->(a:Area {areaCode: 'A1'})
RETURN o.name, c.type LIMIT 50
```

**Why include `{chat_history}`?**
A follow-up question like "Show me the officer's badge numbers" relies on the model knowing what
"the officer" refers to from the previous turn.

---

### 3. `cypher_validate` — Cypher Validator

**Purpose:** Score a generated Cypher query and optionally provide a corrected version.

**System message (condensed):**
> Validate the Cypher against the schema. Return JSON with `score` (0–1), `issues` (list), and
> `corrected_cypher` (string).

**Scoring guidance:**

| Score range   | Meaning |
|---------------|---------|
| `>= 0.7`      | Likely correct — safe to execute. |
| `0.4 – 0.69`  | Fixable — provide `corrected_cypher`. |
| `< 0.4`       | Fundamentally wrong or ambiguous — do not attempt to run. |

**What it checks:**
- Wrong relationship direction (e.g. `(Crime)-[:OCCURRED_AT]->(Officer)`)
- Wrong or non-existent labels/property names/case sensitivity
- Missing `RETURN` clause
- Overly broad queries without `LIMIT`
- Mismatch between the query and the user's intent

**Human message variables:**
| Variable   | Filled by |
|------------|-----------|
| `{schema}` | `Neo4jGraph.schema` |
| `{question}` | The original user question |
| `{cypher}` | The Cypher string to validate |

**Note:** `{chat_history}` is intentionally **not** included here. Validation is a structural
check against the schema; it does not need conversational context.

**Expected LLM output:**
```json
{
  "score": 0.9,
  "issues": [],
  "corrected_cypher": ""
}
```

---

### 4. `answer` — Answer Formatter

**Purpose:** Convert raw Neo4j query results (passed as a JSON string) into a friendly,
human-readable answer.

**System message (condensed):**
> You are a crime investigation assistant. Convert raw results into a clear answer. If results
> are empty, say you couldn't find matching records and suggest how to refine the query.

**Human message variables:**
| Variable        | Filled by |
|-----------------|-----------|
| `{question}`    | The original user question |
| `{results}`     | `json.dumps(serialized_results)` — the Neo4j rows or vector search results |
| `{chat_history}` | Previous turns |

**Why include `{chat_history}` here?**
The answer formatter may need context to correctly use pronouns or refer back to earlier
answers (e.g. "As mentioned before, Officer Smith also investigated…").

---

### 5. `cypher_fix` — Cypher Error Fixer

**Purpose:** Repair a Cypher query that caused a Neo4j runtime exception.

**System message constraints (summarised):**
- Output read-only Cypher only.
- Use only schema-valid labels/relationships/properties.
- Preserve the original user intent.
- Always include an explicit `RETURN`.
- Prefer `LIMIT 50`.

**Human message variables:**
| Variable    | Filled by |
|-------------|-----------|
| `{schema}`  | `Neo4jGraph.schema` |
| `{question}` | The original user question |
| `{cypher}`  | The Cypher that failed |
| `{error}`   | The Neo4j error message string |

**Note:** `{chat_history}` is not included because fixing a runtime error is a purely technical
task; the model only needs the schema, the failed query, and the error message.

---

### 6. `hybrid_filter` — Hybrid Search Graph Filter

**Purpose:** Generate a Cypher query that takes a list of candidate node IDs (from a prior
vector search) and applies additional graph constraints to narrow down the results.

**System message requirements (summarised):**
- Start the query with `WITH $candidate_ids AS candidate_ids`.
- Match candidates with `WHERE elementId(n) IN candidate_ids`.
- Add graph constraints implied by the question.
- Return a small projection with `LIMIT 50`.
- Output only Cypher.

**Human message variables:**
| Variable    | Filled by |
|-------------|-----------|
| `{schema}`  | `Neo4jGraph.schema` |
| `{question}` | The user's question |

**Why must the query start with `WITH $candidate_ids`?**
The caller (`CrimeKGAI.hybrid_search`) passes the candidate node ID list as a query parameter
named `candidate_ids`. Starting with `WITH $candidate_ids AS candidate_ids` brings the parameter
into scope so the rest of the query can reference it.

---

## Why Use `ChatPromptTemplate.from_messages`?

LangChain's `ChatPromptTemplate` separates system and human turn messages, which maps directly
onto the chat model's API (system prompt + user turn). Using `from_messages` instead of a
single string template means:

1. The **system message** role is explicitly set, so the model treats it as authoritative
   instructions rather than part of the conversation.
2. `MessagesPlaceholder("chat_history")` inserts the full conversation history at the correct
   position (between system and the current human message) without any manual string formatting.
3. Variables (`{schema}`, `{question}`, etc.) are filled in at call time via
   `prompt.format_messages(schema=…, question=…)`, keeping the template reusable.
