# HOW IT WORKS — `embedding_setup.py`

This file implements a one-shot data-preparation pipeline: it reads nodes from Neo4j that are
missing vector embeddings, generates those embeddings with Gemini, and writes them back to the
database. It also creates the Neo4j vector index if it does not already exist.

You run this script once (or whenever new unembedded nodes are added to the database) before
starting the main assistant. The assistant's vector search and hybrid search only work correctly
after this pipeline has been run at least once.

---

## `EmbeddingSetupConfig` Dataclass

```python
@dataclass
class EmbeddingSetupConfig:
    index_name: str
    node_label: str
    text_props: List[str]
    embedding_prop: str
    limit: int
    batch_size: int
```

A typed configuration object that carries all runtime settings. Using a dataclass rather than
passing many arguments around means each function only needs to accept one parameter, and it is
easy to see exactly what can be configured.

| Field           | Default (env var)                         | Description |
|-----------------|-------------------------------------------|-------------|
| `index_name`    | `crime_embeddings` (`NEO4J_VECTOR_INDEX`) | Name of the Neo4j vector index to create or reuse. |
| `node_label`    | `Crime` (`NEO4J_VECTOR_LABEL`)            | Which node label to embed. |
| `text_props`    | `["type", "last_outcome"]` (`NEO4J_VECTOR_TEXT_PROPS`) | Node properties whose values are concatenated to form the text that is embedded. |
| `embedding_prop`| `embedding` (`NEO4J_VECTOR_EMBEDDING_PROP`) | The property name on each node where the vector will be stored. |
| `limit`         | `2000` (`NEO4J_VECTOR_EMBED_LIMIT`)       | Maximum number of nodes to process in one run. |
| `batch_size`    | `64` (`NEO4J_VECTOR_EMBED_BATCH`)         | Number of texts sent to the embedding API in one request. |

---

## `parse_args() -> EmbeddingSetupConfig`

### Purpose
Parse command-line arguments and build an `EmbeddingSetupConfig`. Environment variables act as
defaults so the script works with no arguments if the variables are set in `.env`.

### How It Works
1. Creates an `argparse.ArgumentParser` with one argument for each config field.
2. Each argument's `default` is read from the corresponding environment variable (via
   `os.getenv(…)`) with a hard-coded fallback if the variable is also absent.
3. After parsing, splits the `--text-props` comma-separated string into a list and strips
   whitespace from each part.
4. Raises `SystemExit` if the resulting list is empty (at least one text property is required
   to generate meaningful embeddings).
5. Returns an `EmbeddingSetupConfig` instance.

### Why Accept Both CLI Args and Env Vars?
- **Env vars** are convenient in Docker/CI environments where secrets are injected via the
  environment.
- **CLI args** are convenient for one-off runs where you want to change one setting without
  editing `.env`.
CLI args take precedence because `argparse` overrides `os.getenv` defaults.

### Inputs
None (reads from `sys.argv`).
### Output
`EmbeddingSetupConfig`

---

## `ensure_required_env() -> None`

### Purpose
Validate that the four credentials needed to connect to Neo4j and Gemini are present before any
network call is made.

### How It Works
Checks `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `GEMINI_API_KEY` with
`os.getenv`. Raises `SystemExit` with a clear message listing only the missing variables.

### Why Call This Separately from `parse_args`?
`parse_args` handles the *shape* of the configuration (index name, batch size, etc.).
`ensure_required_env` handles *secrets* (credentials). Separating them makes each function
easier to test and reason about.

### Inputs / Output
None / None (raises `SystemExit` on failure).

---

## `build_driver() -> Driver`

### Purpose
Create a raw Neo4j `Driver` (the lower-level official Python driver, as opposed to LangChain's
`Neo4jGraph`) for use in the embedding pipeline.

### Why Use the Raw Driver Here Instead of `Neo4jGraph`?
The embedding pipeline needs to run large `UNWIND … SET` write queries to store embeddings on
nodes. `Neo4jGraph` is a read-oriented wrapper designed for `MATCH … RETURN` queries. The raw
`Driver` gives full access to transactions and is more appropriate for bulk writes.

### Inputs
None (reads from `os.environ`).
### Output
`neo4j.Driver` — an authenticated connection to the Neo4j instance.

---

## `build_embeddings() -> GoogleGenerativeAIEmbeddings`

### Purpose
Instantiate the Gemini embedding model.

### How It Works
Creates `GoogleGenerativeAIEmbeddings` with the hard-coded model `"gemini-embedding-001"`.
This model produces 768-dimensional cosine-normalised vectors — the same dimension as the
vector index created by `ensure_vector_index`.

Prints `"Embedding model initialized."` to confirm the model is ready before any API calls.

### Inputs
None (reads `GEMINI_API_KEY` from `os.environ`).
### Output
`GoogleGenerativeAIEmbeddings`

---

## `ensure_vector_index(session: Session, config: EmbeddingSetupConfig) -> None`

### Purpose
Create the Neo4j vector index if it does not already exist.

### How It Works
Runs the following Cypher (with values interpolated from `config`):
```cypher
CREATE VECTOR INDEX <index_name> IF NOT EXISTS
FOR (n:<node_label>) ON (n.<embedding_prop>)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }
```

The `IF NOT EXISTS` clause makes this operation idempotent — running it multiple times has the
same effect as running it once. `vector.dimensions: 768` must match the output size of
`gemini-embedding-001`. `cosine` similarity is appropriate for text embeddings because it
measures the angle between vectors regardless of their magnitude.

### Inputs
| Parameter | Description |
|-----------|-------------|
| `session` | An open Neo4j `Session`. |
| `config`  | `EmbeddingSetupConfig` (uses `index_name`, `node_label`, `embedding_prop`). |

### Output
None (runs DDL in Neo4j).

---

## `fetch_nodes_to_embed(session, config) -> List[Tuple[str, str]]`

### Purpose
Query Neo4j for all nodes of the configured label that do not yet have an embedding, up to
`config.limit`, and build a `(node_id, text_to_embed)` list.

### How It Works
1. Runs:
   ```cypher
   MATCH (n:<node_label>)
   WHERE n.<embedding_prop> IS NULL
   RETURN elementId(n) AS eid, properties(n) AS props
   LIMIT $limit
   ```
   `IS NULL` ensures only nodes that have not been embedded yet are returned — re-running the
   script after a partial failure will pick up where it left off.

2. For each row:
   - Extracts the `elementId` (a string unique identifier for the node).
   - For each property name in `config.text_props`, formats `"property_name: value"` and
     appends to a parts list. Properties that are `None` are skipped.
   - Joins all parts with `" | "` to produce the embedding text.
   - Falls back to `str(props)` if all configured properties were `None`.

3. Returns a list of `(elementId, text)` tuples.

### Inputs
| Parameter | Description |
|-----------|-------------|
| `session` | An open Neo4j `Session`. |
| `config`  | `EmbeddingSetupConfig` (uses `node_label`, `embedding_prop`, `text_props`, `limit`). |

### Output
`List[Tuple[str, str]]` — pairs of `(neo4j_element_id, text_to_embed)`.

### Why `elementId` Instead of a Property Like `crimeId`?
`elementId` is guaranteed to be unique and stable for every node regardless of what properties
the node has. Using it as the key means this function works for any node label, not just `:Crime`.

---

## `update_embeddings(session, embeddings, config, nodes) -> int`

### Purpose
Generate embedding vectors for all the collected `(id, text)` pairs and write them back to
Neo4j in batches.

### How It Works

```
For each batch of config.batch_size nodes:
  1. Extract the text strings from the batch.
  2. Call embeddings.embed_documents(texts) → list of 768-dim float vectors.
  3. Build a payload list: [{"eid": id, "embedding": vector}, ...].
  4. Run Cypher:
       UNWIND $rows AS row
       MATCH (n) WHERE elementId(n) = row.eid
       SET n.<embedding_prop> = row.embedding
  5. Increment the running count and print progress.
```

**Why batch?**
The Gemini embedding API has rate limits and per-request token limits. Batching 64 texts per
call strikes a balance between throughput and staying within API limits. The batch size is
configurable via `--batch-size` / `NEO4J_VECTOR_EMBED_BATCH`.

**Why `UNWIND … MATCH … SET` instead of individual queries?**
A single `UNWIND` query writes an entire batch in one round-trip to Neo4j, which is far more
efficient than sending one `SET` query per node.

### Inputs
| Parameter    | Description |
|--------------|-------------|
| `session`    | An open Neo4j `Session`. |
| `embeddings` | `GoogleGenerativeAIEmbeddings` instance. |
| `config`     | `EmbeddingSetupConfig` (uses `batch_size`, `embedding_prop`). |
| `nodes`      | The list returned by `fetch_nodes_to_embed`. |

### Output
`int` — total number of nodes successfully updated.

---

## `run_embedding_setup(config: EmbeddingSetupConfig) -> None`

### Purpose
Orchestrate the full pipeline: load env, validate credentials, initialise models, connect to
the database, create the index, embed nodes, and close the connection.

### How It Works

```
load_dotenv()
ensure_required_env()
embeddings = build_embeddings()
driver = build_driver()
try:
    with driver.session() as session:
        ensure_vector_index(session, config)
        nodes = fetch_nodes_to_embed(session, config)
        if not nodes:
            print("Nothing to do.")
            return
        updated = update_embeddings(session, embeddings, config, nodes)
        print("Done. … embedded=<updated>")
finally:
    driver.close()
```

The `try/finally` block guarantees that `driver.close()` is always called, even if an exception
occurs mid-pipeline, so network connections are not leaked.

### Inputs
`config` — `EmbeddingSetupConfig` produced by `parse_args`.
### Output
None (prints progress and summary to stdout, writes embeddings to Neo4j).
