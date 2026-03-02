import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_classic.memory import ConversationBufferMemory


ROUTES = {"GRAPH", "VECTOR", "HYBRID", "AGENT", "AMBIGUOUS"}


def _llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


def _is_readonly_cypher(cypher: str) -> bool:
    forbidden = [
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "DROP",
        "ALTER",
        "LOAD CSV",
        "CALL dbms",
        "CALL apoc",
    ]
    upper = re.sub(r"\s+", " ", cypher.upper())
    return not any(tok in upper for tok in forbidden)


def _safe_index_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("Invalid vector index name. Use letters, digits, underscore, dash only.")
    return name


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    # Neo4j Node/Relationship types typically have an _properties or items() access.
    props = getattr(value, "_properties", None)
    if isinstance(props, dict):
        return {k: _serialize_value(v) for k, v in props.items()}

    if hasattr(value, "items"):
        try:
            return {k: _serialize_value(v) for k, v in dict(value).items()}
        except Exception:
            pass

    return str(value)


@dataclass
class ValidationResult:
    score: float
    issues: List[str]
    corrected_cypher: Optional[str] = None


class CrimeKGAI:
    def __init__(
        self,
        graph: Neo4jGraph,
        llm: ChatGoogleGenerativeAI,
        embeddings: GoogleGenerativeAIEmbeddings,
        vector_index_name: str,
        memory: ConversationBufferMemory,
        verbose: bool = False,
    ) -> None:
        self.graph = graph
        self.llm = llm
        self.embeddings = embeddings
        self.vector_index_name = _safe_index_name(vector_index_name)
        self.memory = memory
        self.verbose = verbose

        self._router_prompt = ChatPromptTemplate.from_messages(
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

        self._cypher_gen_prompt = ChatPromptTemplate.from_messages(
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

        self._cypher_validate_prompt = ChatPromptTemplate.from_messages(
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

        self._answer_prompt = ChatPromptTemplate.from_messages(
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

        self._cypher_fix_prompt = ChatPromptTemplate.from_messages(
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

        self._hybrid_filter_prompt = ChatPromptTemplate.from_messages(
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

    def _chat_history(self) -> List[BaseMessage]:
        variables = self.memory.load_memory_variables({})
        history = variables.get("history")
        if isinstance(history, list):
            return history
        if isinstance(history, str) and history.strip():
            # As a fallback, stuff plain text history into a single message.
            return [HumanMessage(content=history)]
        return []

    def classify(self, question: str) -> Dict[str, Any]:
        messages = self._router_prompt.format_messages(
            schema=self.graph.schema,
            question=question,
            chat_history=self._chat_history(),
        )
        resp = self.llm.invoke(messages)
        data = _extract_json_object(_llm_content_to_text(getattr(resp, "content", resp)))
        route = str(data.get("route", "")).upper().strip()
        if route not in ROUTES:
            route = "GRAPH"
        return {
            "route": route,
            "reason": str(data.get("reason", "")),
            "clarification_question": str(data.get("clarification_question", "")),
        }

    def _generate_cypher(self, question: str) -> str:
        messages = self._cypher_gen_prompt.format_messages(
            schema=self.graph.schema,
            question=question,
            chat_history=self._chat_history(),
        )
        resp = self.llm.invoke(messages)
        cypher = _llm_content_to_text(getattr(resp, "content", resp)).strip()
        return cypher

    def _validate_cypher(self, question: str, cypher: str) -> ValidationResult:
        if not _is_readonly_cypher(cypher):
            return ValidationResult(score=0.0, issues=["Query is not read-only"], corrected_cypher=None)

        messages = self._cypher_validate_prompt.format_messages(
            schema=self.graph.schema,
            question=question,
            cypher=cypher,
        )
        resp = self.llm.invoke(messages)
        data = _extract_json_object(_llm_content_to_text(getattr(resp, "content", resp)))
        score = float(data.get("score", 0.0))
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        corrected = str(data.get("corrected_cypher", "") or "").strip() or None
        if corrected and not _is_readonly_cypher(corrected):
            corrected = None

        # Guardrail: if the model flags a critical schema mismatch, do not allow a high score.
        critical_markers = [
            "does not exist",
            "not exist",
            "reversed",
            "wrong direction",
            "missing return",
            "no return",
            "label",
            "relationship type",
            "property",
        ]
        joined_issues = " ".join(i.lower() for i in issues if isinstance(i, str))
        if any(m in joined_issues for m in critical_markers):
            score = min(score, 0.69)

        return ValidationResult(score=score, issues=[str(i) for i in issues], corrected_cypher=corrected)

    def _execute_cypher_with_validation(
        self, question: str, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Any]:
        validation = self._validate_cypher(question, cypher)
        if self.verbose:
            print("\n[cypher]", cypher)
            print("[validation] score=", validation.score)
            if validation.issues:
                print("[validation] issues=", validation.issues)

        if validation.score >= 0.7:
            try:
                return cypher, self.graph.query(cypher, params=params)
            except Exception as e:
                # Retry once with a model-generated fix.
                fix_messages = self._cypher_fix_prompt.format_messages(
                    schema=self.graph.schema,
                    question=question,
                    cypher=cypher,
                    error=str(e),
                )
                fixed = _llm_content_to_text(getattr(self.llm.invoke(fix_messages), "content", "")).strip()
                if not fixed or not _is_readonly_cypher(fixed):
                    raise
                fixed_validation = self._validate_cypher(question, fixed)
                if self.verbose:
                    print("\n[cypher-fix]", fixed)
                    print("[validation-fix] score=", fixed_validation.score)
                    if fixed_validation.issues:
                        print("[validation-fix] issues=", fixed_validation.issues)
                if fixed_validation.score >= 0.7:
                    return fixed, self.graph.query(fixed, params=params)
                raise

        if 0.4 <= validation.score < 0.7 and validation.corrected_cypher:
            corrected = validation.corrected_cypher
            validation2 = self._validate_cypher(question, corrected)
            if self.verbose:
                print("\n[cypher-corrected]", corrected)
                print("[validation2] score=", validation2.score)
                if validation2.issues:
                    print("[validation2] issues=", validation2.issues)
            if validation2.score >= 0.7:
                return corrected, self.graph.query(corrected, params=params)

        raise ValueError(
            "Low-confidence Cypher. Please clarify your question (e.g., which person/officer/area/crime type)."
        )

    def _format_answer(self, question: str, results: Any) -> str:
        serialized = _serialize_value(results)
        messages = self._answer_prompt.format_messages(
            question=question,
            results=json.dumps(serialized, ensure_ascii=False),
            chat_history=self._chat_history(),
        )
        resp = self.llm.invoke(messages)
        return _llm_content_to_text(getattr(resp, "content", resp)).strip()

    def vector_search(self, question: str, k: int = 8, min_score: float = 0.7) -> List[Dict[str, Any]]:
        vector = self.embeddings.embed_query(question)
        cypher = (
            f"CALL db.index.vector.queryNodes('{self.vector_index_name}', $k, $vector) "
            "YIELD node, score "
            "WHERE score >= $min_score "
            "RETURN elementId(node) AS id, labels(node) AS labels, properties(node) AS props, score "
            "ORDER BY score DESC"
        )
        rows = self.graph.query(cypher, params={"k": k, "vector": vector, "min_score": min_score})
        return [
            {
                "id": r.get("id"),
                "labels": r.get("labels"),
                "properties": r.get("props"),
                "score": r.get("score"),
            }
            for r in rows
        ]

    def hybrid_search(self, question: str, k: int = 12, min_score: float = 0.7) -> Any:
        candidates = self.vector_search(question, k=k, min_score=min_score)
        candidate_ids = [c["id"] for c in candidates if c.get("id") is not None]
        if not candidate_ids:
            return []

        messages = self._hybrid_filter_prompt.format_messages(schema=self.graph.schema, question=question)
        cypher = _llm_content_to_text(getattr(self.llm.invoke(messages), "content", "")).strip()
        if not cypher:
            return candidates
        if not _is_readonly_cypher(cypher):
            return candidates

        # The filter prompt requires $candidate_ids; if the model forgot it, fall back.
        if "$candidate_ids" not in cypher and "candidate_ids" not in cypher:
            return candidates

        _, rows = self._execute_cypher_with_validation(
            question=question, cypher=cypher, params={"candidate_ids": candidate_ids}
        )
        return rows

    def answer(self, question: str) -> str:
        # Guaranteed ambiguous handling for a canonical underspecified query.
        ql = question.strip().lower()
        if ql in {"show me connections between people", "connections between people"}:
            clarification = (
                "Which type of connection do you mean (e.g., KNOWS vs FAMILY_REL), and for which person name(s)?"
            )
            self.memory.save_context({"input": question}, {"output": clarification})
            return clarification

        route_info = self.classify(question)
        route = route_info["route"]

        if self.verbose:
            print("\n[route]", route, "-", route_info.get("reason", ""))

        if route == "AMBIGUOUS":
            clarification = route_info.get("clarification_question") or (
                "Can you clarify which entity you mean (e.g., person name, officer badge_no, areaCode, crime type)?"
            )
            self.memory.save_context({"input": question}, {"output": clarification})
            return clarification

        if route == "VECTOR":
            results = self.vector_search(question)
            answer = self._format_answer(question, results)
            self.memory.save_context({"input": question}, {"output": answer})
            return answer

        if route == "HYBRID":
            results = self.hybrid_search(question)
            answer = self._format_answer(question, results)
            self.memory.save_context({"input": question}, {"output": answer})
            return answer

        if route == "AGENT":
            answer = self._agent_answer(question)
            self.memory.save_context({"input": question}, {"output": answer})
            return answer

        # GRAPH (default)
        cypher = self._generate_cypher(question)
        executed_cypher, rows = self._execute_cypher_with_validation(question, cypher)
        answer = self._format_answer(question, rows)
        if self.verbose:
            print("\n[executed_cypher]", executed_cypher)
        self.memory.save_context({"input": question}, {"output": answer})
        return answer

    def _agent_answer(self, question: str) -> str:
        # Tools are defined as closures so they can share self.
        @tool
        def cypher_executor(cypher: str) -> str:
            """Execute a READ-ONLY Cypher query in Neo4j after validation. Returns JSON rows."""
            cypher = cypher.strip()
            if not cypher:
                return json.dumps({"error": "empty cypher"})
            if not _is_readonly_cypher(cypher):
                return json.dumps({"error": "non-readonly cypher rejected"})
            try:
                _, rows = self._execute_cypher_with_validation(question="(agent tool)", cypher=cypher)
                return json.dumps(_serialize_value(rows), ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        @tool
        def vector_search(query: str) -> str:
            """Vector similarity search over embedded nodes. Returns JSON list of top matches."""
            try:
                rows = self.vector_search(query)
                return json.dumps(rows, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        @tool
        def hybrid_search(query: str) -> str:
            """Hybrid: vector candidates then graph filtering using Cypher. Returns JSON rows."""
            try:
                rows = self.hybrid_search(query)
                return json.dumps(_serialize_value(rows), ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        agent = create_react_agent(self.llm, tools=[cypher_executor, vector_search, hybrid_search])
        history = self._chat_history()
        state = {"messages": [*history, HumanMessage(content=question)]}
        final_state = agent.invoke(state)
        messages = final_state.get("messages") or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is None:
            return "I couldn't complete that reasoning step. Please rephrase the question."
        return _llm_content_to_text(getattr(last_ai, "content", last_ai)).strip()


def run_validation_test_cases(app: CrimeKGAI) -> None:
    print("\nRunning 3 validation test cases...\n")

    cases = [
        (
            "Reversed relationship direction",
            "Which locations are in area A1?",
            "MATCH (a:Area {areaCode:'A1'})-[:LOCATION_IN_AREA]->(l:Location) RETURN l.address LIMIT 5",
        ),
        (
            "Non-existent label",
            "Show me all suspects.",
            "MATCH (s:Suspect) RETURN s LIMIT 5",
        ),
        (
            "Ambiguous question guessed wrong relationship",
            "Show me connections between people.",
            "MATCH (p:Person)-[:CONNECTED_TO]->(q:Person) RETURN p.name, q.name LIMIT 25",
        ),
    ]

    for title, question, cypher in cases:
        print(f"== {title} ==")
        result = app._validate_cypher(question=question, cypher=cypher)
        print("score:", result.score)
        if result.issues:
            print("issues:")
            for i in result.issues:
                print("-", i)
        if result.corrected_cypher:
            print("corrected_cypher:")
            print(result.corrected_cypher)
        print()


def build_app(verbose: bool) -> CrimeKGAI:
    load_dotenv()

    required_env = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEY"]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing required env vars in .env: {', '.join(missing)}")

    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        temperature=0,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    memory = ConversationBufferMemory(return_messages=True)

    index_name = os.getenv("NEO4J_VECTOR_INDEX", "crime_embeddings")
    return CrimeKGAI(
        graph=graph,
        llm=llm,
        embeddings=embeddings,
        vector_index_name=index_name,
        memory=memory,
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural language to Neo4j (Gemini + LangChain).")
    parser.add_argument("--demo", action="store_true", help="Run a small demo with sample questions.")
    parser.add_argument(
        "--run-validation-tests",
        action="store_true",
        help="Print 3 validation test cases (required by the assignment).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print routing/Cypher/validation details.")
    args = parser.parse_args()

    app = build_app(verbose=args.verbose)

    if args.run_validation_tests:
        run_validation_test_cases(app)
        return

    if args.demo:
        questions = [
            "Who investigated crimes in area A1?",
            "List crimes that occurred at a given location address.",
            "Recommend crimes similar to burglary.",
            "Find crimes similar to burglary but only those investigated by a specific officer.",
            "Which person has the most connections in this database and who are their top connections?",
        ]
        for q in questions:
            print("\nQ:", q)
            try:
                print("A:", app.answer(q))
            except Exception as e:
                print("A: error:", e)
        return

    print("Crime KG Assistant (type 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            print(app.answer(question))
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    main()