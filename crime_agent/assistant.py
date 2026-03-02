import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_classic.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import create_react_agent

from .prompts import build_prompt_set
from .utils import (
    extract_json_object,
    is_readonly_cypher,
    llm_content_to_text,
    safe_index_name,
    serialize_value,
)


ROUTES = {"GRAPH", "VECTOR", "HYBRID", "AGENT", "AMBIGUOUS"}


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
        self.vector_index_name = safe_index_name(vector_index_name)
        self.memory = memory
        self.verbose = verbose

        prompts = build_prompt_set()
        self._router_prompt = prompts.router
        self._cypher_gen_prompt = prompts.cypher_gen
        self._cypher_validate_prompt = prompts.cypher_validate
        self._answer_prompt = prompts.answer
        self._cypher_fix_prompt = prompts.cypher_fix
        self._hybrid_filter_prompt = prompts.hybrid_filter

    def _chat_history(self) -> List[BaseMessage]:
        variables = self.memory.load_memory_variables({})
        history = variables.get("history")
        if isinstance(history, list):
            return history
        if isinstance(history, str) and history.strip():
            return [HumanMessage(content=history)]
        return []

    def classify(self, question: str) -> Dict[str, Any]:
        messages = self._router_prompt.format_messages(
            schema=self.graph.schema,
            question=question,
            chat_history=self._chat_history(),
        )
        resp = self.llm.invoke(messages)
        data = extract_json_object(llm_content_to_text(getattr(resp, "content", resp)))
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
        return llm_content_to_text(getattr(resp, "content", resp)).strip()

    def _validate_cypher(self, question: str, cypher: str) -> ValidationResult:
        if not is_readonly_cypher(cypher):
            return ValidationResult(score=0.0, issues=["Query is not read-only"], corrected_cypher=None)

        messages = self._cypher_validate_prompt.format_messages(
            schema=self.graph.schema,
            question=question,
            cypher=cypher,
        )
        resp = self.llm.invoke(messages)
        data = extract_json_object(llm_content_to_text(getattr(resp, "content", resp)))
        score = float(data.get("score", 0.0))
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        corrected = str(data.get("corrected_cypher", "") or "").strip() or None
        if corrected and not is_readonly_cypher(corrected):
            corrected = None

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
        if any(marker in joined_issues for marker in critical_markers):
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
                fix_messages = self._cypher_fix_prompt.format_messages(
                    schema=self.graph.schema,
                    question=question,
                    cypher=cypher,
                    error=str(e),
                )
                fixed = llm_content_to_text(getattr(self.llm.invoke(fix_messages), "content", "")).strip()
                if not fixed or not is_readonly_cypher(fixed):
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
        serialized = serialize_value(results)
        messages = self._answer_prompt.format_messages(
            question=question,
            results=json.dumps(serialized, ensure_ascii=False),
            chat_history=self._chat_history(),
        )
        resp = self.llm.invoke(messages)
        return llm_content_to_text(getattr(resp, "content", resp)).strip()

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
                "id": row.get("id"),
                "labels": row.get("labels"),
                "properties": row.get("props"),
                "score": row.get("score"),
            }
            for row in rows
        ]

    def hybrid_search(self, question: str, k: int = 12, min_score: float = 0.7) -> Any:
        candidates = self.vector_search(question, k=k, min_score=min_score)
        candidate_ids = [c["id"] for c in candidates if c.get("id") is not None]
        if not candidate_ids:
            return []

        messages = self._hybrid_filter_prompt.format_messages(schema=self.graph.schema, question=question)
        cypher = llm_content_to_text(getattr(self.llm.invoke(messages), "content", "")).strip()
        if not cypher:
            return candidates
        if not is_readonly_cypher(cypher):
            return candidates
        if "$candidate_ids" not in cypher and "candidate_ids" not in cypher:
            return candidates

        _, rows = self._execute_cypher_with_validation(
            question=question, cypher=cypher, params={"candidate_ids": candidate_ids}
        )
        return rows

    def answer(self, question: str) -> str:
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

        cypher = self._generate_cypher(question)
        executed_cypher, rows = self._execute_cypher_with_validation(question, cypher)
        answer = self._format_answer(question, rows)
        if self.verbose:
            print("\n[executed_cypher]", executed_cypher)
        self.memory.save_context({"input": question}, {"output": answer})
        return answer

    def _agent_answer(self, question: str) -> str:
        @tool
        def cypher_executor(cypher: str) -> str:
            """Execute a READ-ONLY Cypher query in Neo4j after validation. Returns JSON rows."""
            cypher = cypher.strip()
            if not cypher:
                return json.dumps({"error": "empty cypher"})
            if not is_readonly_cypher(cypher):
                return json.dumps({"error": "non-readonly cypher rejected"})
            try:
                _, rows = self._execute_cypher_with_validation(question="(agent tool)", cypher=cypher)
                return json.dumps(serialize_value(rows), ensure_ascii=False)
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
                return json.dumps(serialize_value(rows), ensure_ascii=False)
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
        return llm_content_to_text(getattr(last_ai, "content", last_ai)).strip()