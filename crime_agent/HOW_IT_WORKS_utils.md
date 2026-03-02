# HOW IT WORKS — `utils.py`

This file contains five pure-utility functions that are shared across every other module in the
`crime_agent` package. None of them call the LLM or the database; they are small, stateless helpers
that handle data-type conversions and safety guards.

---

## 1. `llm_content_to_text(content: Any) -> str`

### Purpose
Every time the code calls `llm.invoke(messages)` it gets back a LangChain `AIMessage` object.
The actual text lives in `AIMessage.content`, but that field is typed as `str | list[...]` because
some models return a list of "content parts" (e.g. text blocks + tool-call blocks) instead of a
plain string. This helper normalises whatever shape comes back into a single plain string.

### How It Works
1. If `content` is already a `str`, return it unchanged.
2. If `content` is a `list`, iterate over every item:
   - If the item is a `dict` that has a `"text"` key whose value is a `str`, collect that text.
   - Otherwise fall back to `str(item)`.
   Join all collected pieces with newlines and return the result.
3. For anything else (an unexpected type), call `str(content)` as a last resort.

### Inputs
| Parameter | Type  | Description                                    |
|-----------|-------|------------------------------------------------|
| `content` | `Any` | The `.content` attribute of an `AIMessage`, or any value that should be turned into text. |

### Output
`str` — a single string representing the full text of the LLM response.

### Why This Function Exists
Without this helper every call site would need an `if isinstance(content, list)` guard. Having one
authoritative place to handle the conversion means that if LangChain changes the shape of
`content` in a future version, only this function needs to be updated.

---

## 2. `extract_json_object(text: str) -> Dict[str, Any]`

### Purpose
The router and validator prompts ask the LLM to output a JSON object. In practice, LLMs frequently
wrap their JSON inside a markdown fenced code block (`` ```json … ``` ``). This function strips
that wrapper and then extracts exactly one JSON object from the remaining text.

### How It Works
1. Strip leading/trailing whitespace from `text`.
2. If the text starts with `` ``` ``, apply a regex to remove the opening fence
   (`` ```json``, `` ```JSON``, `` ``` `` etc.) and the closing `` ``` ``.
3. Find the position of the first `{` and the last `}` in the cleaned text.
4. If either is missing, or if `start >= end`, raise a `ValueError` with a helpful excerpt of
   the raw output so the developer can see what the model actually returned.
5. Call `json.loads` on the substring `text[start : end + 1]` and return the resulting `dict`.

### Inputs
| Parameter | Type  | Description |
|-----------|-------|-------------|
| `text`    | `str` | Raw LLM output that is expected to contain exactly one JSON object. |

### Output
`Dict[str, Any]` — the parsed JSON object.

### Why This Function Exists
`json.loads` by itself would fail whenever the model adds prose around the JSON or wraps it in
a code fence. This function makes JSON extraction robust to the two most common LLM formatting
habits while still raising a clear error when the output is completely non-JSON.

---

## 3. `is_readonly_cypher(cypher: str) -> bool`

### Purpose
The assistant must never allow write operations to be executed against the Neo4j database. This
function acts as a last-resort safety gate that rejects any Cypher string containing a
data-mutation keyword.

### How It Works
1. Defines a list of **forbidden tokens**:
   `CREATE`, `MERGE`, `DELETE`, `DETACH`, `SET`, `DROP`, `ALTER`, `LOAD CSV`,
   `CALL dbms`, `CALL apoc`.
2. Uses `re.sub(r"\s+", " ", cypher.upper())` to collapse all whitespace and upper-case the query.
   This prevents bypassing the check with unusual spacing like `C R E A T E`.
3. Returns `True` if **none** of the forbidden tokens appear in the normalised string, `False`
   otherwise.

### Inputs
| Parameter | Type  | Description |
|-----------|-------|-------------|
| `cypher`  | `str` | Any Cypher query string to check. |

### Output
`bool` — `True` means the query contains only read operations and is safe to run. `False` means it
contains at least one write/admin keyword and must be rejected.

### Why This Function Exists
The LLM is instructed to produce only `MATCH … RETURN` queries, but prompt injection or model
hallucination could still produce a destructive query. This function is called at every point
where a Cypher string could reach the database, providing defence-in-depth on top of the prompt
instructions.

---

## 4. `safe_index_name(name: str) -> str`

### Purpose
The Neo4j vector index name is supplied by the operator (via an environment variable). Before that
name is interpolated into a raw Cypher string, this function validates it to prevent Cypher
injection attacks.

### How It Works
1. Uses `re.fullmatch(r"[A-Za-z0-9_\-]+", name)` to assert that the name consists **only** of
   alphanumeric characters, underscores, and hyphens.
2. If the pattern does not match (e.g. the name contains spaces, quotes, or semicolons), raises a
   `ValueError` with a descriptive message.
3. If the pattern matches, returns `name` unchanged.

### Inputs
| Parameter | Type  | Description |
|-----------|-------|-------------|
| `name`    | `str` | The candidate vector index name from configuration. |

### Output
`str` — the same `name` string, guaranteed to be safe for direct embedding in Cypher.

### Why This Function Exists
The vector index name is used directly inside an f-string Cypher template
(`f"CALL db.index.vector.queryNodes('{self.vector_index_name}', …)"`). Without this check, a
malicious or misconfigured name such as `x', 1, [1]) CALL apoc.…//` could alter the query's
meaning. Validating the name at construction time (in `CrimeKGAI.__init__`) means it only needs
to be checked once.

---

## 5. `serialize_value(value: Any) -> Any`

### Purpose
Neo4j returns query results as Python objects that can contain Neo4j-specific types (e.g.
`Node`, `Relationship`, `Path`) which are not natively serialisable by `json.dumps`. This
recursive function converts any Neo4j result value into a plain Python value that `json.dumps`
can handle.

### How It Works
The function uses a series of `isinstance` checks applied from most-specific to most-general:

1. **Primitives** (`str`, `int`, `float`, `bool`) and `None` — returned as-is.
2. **`list`** — each element is recursively serialised.
3. **`dict`** — each value is recursively serialised (keys are assumed to already be strings).
4. **Neo4j `Node` / `Relationship`** — these objects expose their properties through a
   `_properties` attribute that is a plain `dict`; serialise that dict.
5. **Any other mapping** — if the object has an `items()` method, convert to a `dict` via
   `dict(value)` and serialise that. This handles `neo4j.Record` and similar types.
6. **Fallback** — `str(value)` to ensure serialisation always succeeds.

### Inputs
| Parameter | Type  | Description |
|-----------|-------|-------------|
| `value`   | `Any` | A single value from a Neo4j query result row. Can be nested. |

### Output
`Any` — a plain Python value (`str`, `int`, `float`, `bool`, `None`, `list`, or `dict`) that is
safe to pass to `json.dumps` or to send to the LLM as part of a prompt.

### Why This Function Exists
The LLM's answer-formatting step receives query results serialised as JSON. If those results
contained raw Neo4j objects, `json.dumps` would raise a `TypeError`. A single recursive
serialiser handles arbitrarily nested results without requiring any changes at the call sites.
