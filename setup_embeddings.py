import argparse
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create/store Gemini embeddings on Neo4j nodes and create the Neo4j vector index."
    )
    parser.add_argument(
        "--index-name",
        default=os.getenv("NEO4J_VECTOR_INDEX", "crime_embeddings"),
        help="Neo4j vector index name (default: crime_embeddings or NEO4J_VECTOR_INDEX).",
    )
    parser.add_argument(
        "--node-label",
        default=os.getenv("NEO4J_VECTOR_LABEL", "Crime"),
        help="Node label to embed (default: Crime or NEO4J_VECTOR_LABEL).",
    )
    parser.add_argument(
        "--text-props",
        default=os.getenv("NEO4J_VECTOR_TEXT_PROPS", "type,last_outcome"),
        help="Comma-separated properties to embed (default: type,last_outcome).",
    )
    parser.add_argument(
        "--embedding-prop",
        default=os.getenv("NEO4J_VECTOR_EMBEDDING_PROP", "embedding"),
        help="Property to store the vector on each node (default: embedding).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("NEO4J_VECTOR_EMBED_LIMIT", "2000")),
        help="Max number of nodes to embed this run (default: 2000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("NEO4J_VECTOR_EMBED_BATCH", "64")),
        help="Embedding batch size (default: 64).",
    )
    args = parser.parse_args()

    required_env = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEY"]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing required env vars in .env: {', '.join(missing)}")

    text_props = [p.strip() for p in args.text_props.split(",") if p.strip()]
    if not text_props:
        raise SystemExit("--text-props must include at least one property name")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    print("Embedding model initialized.")

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session() as session:
            # Ensure vector index exists.
            create_index_cypher = (
                f"CREATE VECTOR INDEX {args.index_name} IF NOT EXISTS "
                f"FOR (n:{args.node_label}) ON (n.{args.embedding_prop}) "
                "OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }"
            )
            session.run(create_index_cypher)

            fetch_cypher = (
                f"MATCH (n:{args.node_label}) "
                f"WHERE n.{args.embedding_prop} IS NULL "
                "RETURN elementId(n) AS eid, properties(n) AS props "
                "LIMIT $limit"
            )
            rows = list(session.run(fetch_cypher, limit=args.limit))
            if not rows:
                print("No nodes found missing embeddings. Nothing to do.")
                return

            to_embed: list[tuple[str, str]] = []
            for r in rows:
                node_eid = str(r["eid"])
                props = dict(r["props"] or {})
                parts = []
                for p in text_props:
                    v = props.get(p)
                    if v is None:
                        continue
                    parts.append(f"{p}: {v}")
                text = " | ".join(parts).strip() or str(props)
                to_embed.append((node_eid, text))

            print(f"Embedding {len(to_embed)} nodes (limit={args.limit}, batch={args.batch_size})...")

            updated = 0
            for start in range(0, len(to_embed), args.batch_size):
                chunk = to_embed[start : start + args.batch_size]
                texts = [t for _, t in chunk]
                vectors = embeddings.embed_documents(texts)
                payload = [
                    {"eid": node_eid, "embedding": vector}
                    for (node_eid, _), vector in zip(chunk, vectors)
                ]
                update_cypher = (
                    "UNWIND $rows AS row "
                    "MATCH (n) WHERE elementId(n) = row.eid "
                    f"SET n.{args.embedding_prop} = row.embedding"
                )
                session.run(update_cypher, rows=payload)
                updated += len(payload)
                print(f"Updated {updated}/{len(to_embed)}")

            print(
                "Done. Vector index ensured and embeddings stored. "
                f"(label={args.node_label}, index={args.index_name}, embedded={updated})"
            )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
    