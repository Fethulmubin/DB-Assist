# CrimeAgent — Natural Language → Neo4j (Gemini + LangChain)

This project implements the **Intern Assignment: Natural Language to Knowledge Graph Query System**.
It accepts plain-English questions, routes them to the right strategy (**graph traversal**, **vector search**, **hybrid**, or **agent**), validates generated Cypher with **confidence scoring + retry**, executes the query on Neo4j, and returns a clean natural-language answer.

## Requirements Covered

- End-to-end pipeline: **NL → Cypher → Neo4j → NL answer**
- Handles **graph**, **vector**, **hybrid**, and **agent** style questions
- **Validation + confidence scoring** with threshold **0.7** before executing Cypher
- **Auto-correct + retry** for fixable Cypher (score 0.4–0.69)
- Handles **ambiguous** queries by asking for clarification
- **Vector embeddings** stored on Neo4j nodes + **Neo4j vector index**
- **Hybrid search**: vector candidates + graph filtering
- Multi-turn follow-ups via **conversation memory**
- Includes **3 validation test cases** demonstrating query problems

## Setup

### 1) Create `.env`

Create a `.env` file in the project root:

```bash
NEO4J_URI=bolt+s://<your-neo4j-host>:7687
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>
GEMINI_API_KEY=<your-gemini-api-key>

# Optional overrides
GEMINI_MODEL=gemini-flash-latest
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
NEO4J_VECTOR_INDEX=crime_embeddings
```

### 2) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Create embeddings + vector index

By default this embeds up to **2000** `:Crime` nodes per run (incremental; only nodes missing embeddings), using the `type` and `last_outcome` properties.

```bash
python setup_embeddings.py
```

If you want to embed more, rerun it (or increase `--limit`):

```bash
python setup_embeddings.py --limit 5000
```

Optional customization:

```bash
python setup_embeddings.py \
  --node-label Crime \
  --index-name crime_embeddings \
  --text-props type,last_outcome \
  --embedding-prop embedding
```

## Run

### Interactive assistant

```bash
python main.py
```

Add extra logging (shows routing, Cypher, and validation score):

```bash
python main.py --verbose
```

### Demo (shows 5 example questions)

```bash
python main.py --demo
```

### Validation test cases (required by assignment)

```bash
python main.py --run-validation-tests
```

## Example Questions (Copy/Paste)

Graph traversal examples:
- "Which officer investigated the most crimes?"
- "Show crimes that occurred in area A1"
- "List objects involved in crime with id C1"

Vector search examples:
- "Recommend crimes similar to burglary"
- "Find crimes like vehicle theft"

Hybrid examples:
- "Find crimes similar to burglary but only those investigated by officer with badge number 123"
- "Find crimes like robbery but only those that occurred in area A1"

Agent (multi-step) example:
- "Which person has the most connections in this database and who are their top connections?"

Multi-turn memory example:
1) "Who investigated crimes in area A1?"
2) "How many crimes did they investigate?"  (should use context)

## Notes

- The assistant **rejects non-read-only Cypher** (no CREATE/MERGE/DELETE/SET/DROP).
- If the system asks for clarification, give a more specific entity (e.g., `areaCode`, `badge_no`, `crime type`, or a person `name`).
