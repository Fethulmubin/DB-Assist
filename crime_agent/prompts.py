from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@dataclass
class PromptSet:
    router: ChatPromptTemplate
    cypher_gen: ChatPromptTemplate
    cypher_validate: ChatPromptTemplate
    answer: ChatPromptTemplate
    cypher_fix: ChatPromptTemplate
    hybrid_filter: ChatPromptTemplate


def build_prompt_set() -> PromptSet:
    router = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are routing user questions for a Neo4j knowledge-graph assistant.
You MUST output a single JSON object with these keys:
- route: one of GRAPH, VECTOR, HYBRID, AGENT, AMBIGUOUS
- reason: short string
- clarification_question: string (required only when route=AMBIGUOUS; otherwise empty)

Routing rules:
- GRAPH: explicit relationship / factual graph traversal (MATCH/WHERE/RETURN).
- VECTOR: semantic similarity / recommendation / 'similar to' questions.
- HYBRID: both semantic similarity AND graph constraints (e.g., similar X but only those connected to Y).
- AGENT: complex multi-step reasoning requiring multiple queries whose count depends on intermediate results.
- AMBIGUOUS: the question is underspecified and cannot be executed safely.
""".strip(),
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "Schema:\n{schema}\n\nQuestion: {question}\n\nReturn JSON only.",
            ),
        ]
    )

    cypher_gen = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You generate READ-ONLY Cypher queries for Neo4j.
Constraints:
- Use only labels/relationship types/properties that exist in the provided schema.
- Never write data (no CREATE/MERGE/DELETE/SET/DROP).
- Always include an explicit RETURN.
- Prefer LIMIT 50 when the result could be large.
Output ONLY the Cypher query, no markdown.
""".strip(),
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "Schema:\n{schema}\n\nQuestion: {question}\n\nCypher:",
            ),
        ]
    )

    cypher_validate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You validate a generated Cypher query against a Neo4j schema.
Return ONLY a JSON object:
{{
    "score": number from 0 to 1,
    "issues": ["..."],
    "corrected_cypher": "..."  // empty string if no correction
}}

Scoring guidance:
- 0.7+ means likely correct.
- 0.4-0.69 means fixable; provide corrected_cypher.
- <0.4 means fundamentally wrong/ambiguous.

Check for:
- wrong relationship direction
- wrong labels/property names/case
- missing RETURN
- overly broad queries without LIMIT
- not matching the question intent
""".strip(),
            ),
            (
                "human",
                "Schema:\n{schema}\n\nQuestion: {question}\n\nCypher:\n{cypher}\n\nJSON:",
            ),
        ]
    )

    answer = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a crime investigation assistant.
You receive raw Neo4j query results (as JSON-like data). Convert them into a clear, human-friendly answer.
If results are empty, say you couldn't find matching records and suggest how to refine.
""".strip(),
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "Question: {question}\n\nResults: {results}\n\nAnswer:",
            ),
        ]
    )

    cypher_fix = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You fix a Cypher query that failed to execute in Neo4j.
Constraints:
- Output READ-ONLY Cypher only (no CREATE/MERGE/DELETE/SET/DROP).
- Use ONLY labels/relationships/properties from the schema.
- Keep the intent aligned with the user question.
- Always include an explicit RETURN.
- Prefer LIMIT 50 when results could be large.
Output ONLY the fixed Cypher, no markdown.
""".strip(),
            ),
            (
                "human",
                "Schema:\n{schema}\n\nQuestion: {question}\n\nFailed Cypher:\n{cypher}\n\nNeo4j error:\n{error}\n\nFixed Cypher:",
            ),
        ]
    )

    hybrid_filter = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You will write a READ-ONLY Cypher query to FILTER a set of candidate nodes.
Inputs:
- schema
- user question
- candidate node ids list (Neo4j elementIds, strings) in $candidate_ids

Requirements:
- Start your query with: WITH $candidate_ids AS candidate_ids
- Match candidates using elementId(n) IN candidate_ids (use the correct label(s) based on schema).
- Add additional graph constraints implied by the question.
- Return a small, useful projection and LIMIT 50.
- Output ONLY Cypher.
""".strip(),
            ),
            (
                "human",
                "Schema:\n{schema}\n\nQuestion: {question}\n\nCypher:",
            ),
        ]
    )

    return PromptSet(
        router=router,
        cypher_gen=cypher_gen,
        cypher_validate=cypher_validate,
        answer=answer,
        cypher_fix=cypher_fix,
        hybrid_filter=hybrid_filter,
    )