import json
import re
from typing import Any, Dict, List


def llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


def is_readonly_cypher(cypher: str) -> bool:
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
    return not any(token in upper for token in forbidden)


def safe_index_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("Invalid vector index name. Use letters, digits, underscore, dash only.")
    return name


def serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}

    props = getattr(value, "_properties", None)
    if isinstance(props, dict):
        return {k: serialize_value(v) for k, v in props.items()}

    if hasattr(value, "items"):
        try:
            return {k: serialize_value(v) for k, v in dict(value).items()}
        except Exception:
            pass

    return str(value)