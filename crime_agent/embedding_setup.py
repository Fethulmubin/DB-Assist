import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j import Driver, GraphDatabase, Session


@dataclass
class EmbeddingSetupConfig:
    index_name: str
    node_label: str
    text_props: List[str]
    embedding_prop: str
    limit: int
    batch_size: int


def parse_args() -> EmbeddingSetupConfig:
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

    text_props = [value.strip() for value in args.text_props.split(",") if value.strip()]
    if not text_props:
        raise SystemExit("--text-props must include at least one property name")

    return EmbeddingSetupConfig(
        index_name=args.index_name,
        node_label=args.node_label,
        text_props=text_props,
        embedding_prop=args.embedding_prop,
        limit=args.limit,
        batch_size=args.batch_size,
    )


def ensure_required_env() -> None:
    required_env = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEY"]
    missing = [key for key in required_env if not os.getenv(key)]
    if missing:
        raise SystemExit(f"Missing required env vars in .env: {', '.join(missing)}")


def build_driver() -> Driver:
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    print("Embedding model initialized.")
    return embeddings


def ensure_vector_index(session: Session, config: EmbeddingSetupConfig) -> None:
    cypher = (
        f"CREATE VECTOR INDEX {config.index_name} IF NOT EXISTS "
        f"FOR (n:{config.node_label}) ON (n.{config.embedding_prop}) "
        "OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }"
    )
    session.run(cypher)


def fetch_nodes_to_embed(session: Session, config: EmbeddingSetupConfig) -> List[Tuple[str, str]]:
    cypher = (
        f"MATCH (n:{config.node_label}) "
        f"WHERE n.{config.embedding_prop} IS NULL "
        "RETURN elementId(n) AS eid, properties(n) AS props "
        "LIMIT $limit"
    )
    rows = list(session.run(cypher, limit=config.limit))
    output: List[Tuple[str, str]] = []

    for row in rows:
        node_id = str(row["eid"])
        props = dict(row["props"] or {})
        parts: List[str] = []
        for prop in config.text_props:
            value = props.get(prop)
            if value is None:
                continue
            parts.append(f"{prop}: {value}")
        text = " | ".join(parts).strip() or str(props)
        output.append((node_id, text))

    return output


def update_embeddings(
    session: Session,
    embeddings: GoogleGenerativeAIEmbeddings,
    config: EmbeddingSetupConfig,
    nodes: List[Tuple[str, str]],
) -> int:
    print(f"Embedding {len(nodes)} nodes (limit={config.limit}, batch={config.batch_size})...")
    updated = 0

    for start in range(0, len(nodes), config.batch_size):
        chunk = nodes[start : start + config.batch_size]
        texts = [text for _, text in chunk]
        vectors = embeddings.embed_documents(texts)

        payload = [
            {"eid": node_id, "embedding": vector}
            for (node_id, _), vector in zip(chunk, vectors)
        ]

        update_cypher = (
            "UNWIND $rows AS row "
            "MATCH (n) WHERE elementId(n) = row.eid "
            f"SET n.{config.embedding_prop} = row.embedding"
        )
        session.run(update_cypher, rows=payload)
        updated += len(payload)
        print(f"Updated {updated}/{len(nodes)}")

    return updated


def run_embedding_setup(config: EmbeddingSetupConfig) -> None:
    load_dotenv()
    ensure_required_env()

    embeddings = build_embeddings()
    driver = build_driver()

    try:
        with driver.session() as session:
            ensure_vector_index(session, config)
            nodes = fetch_nodes_to_embed(session, config)
            if not nodes:
                print("No nodes found missing embeddings. Nothing to do.")
                return

            updated = update_embeddings(session, embeddings, config, nodes)
            print(
                "Done. Vector index ensured and embeddings stored. "
                f"(label={config.node_label}, index={config.index_name}, embedded={updated})"
            )
    finally:
        driver.close()