import json
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ai.key import get_gemini_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class QueryPlan(BaseModel):
    raw_query: str                  = Field(description="Original user query")
    domain_hint: Optional[str]      = Field(None, description="Subject domain inferred from the query (e.g. 'climate', 'finance', 'medicine') — null if unclear")
    topic: Optional[str]            = Field(None, description="Specific topic or subject within the domain (e.g. 'carbon emissions', 'loan interest rates')")
    entity_focus: Optional[str]     = Field(None, description="Named entity the query is primarily about (a person, organisation, concept, policy, product, etc.)")
    context_attributes: list[str]   = Field(default_factory=list, description="Any contextual constraints or filters mentioned (e.g. 'after 2020', 'in the EU', 'for SMEs')")
    keywords: list[str]             = Field(default_factory=list, description="3-7 important concepts to search for in documents")
    is_multi_hop: bool              = Field(False, description="True only when the query genuinely requires retrieving from two clearly separate topic areas")
    intent: str                     = Field("find_information", description="One of: find_information, explain_concept, compare_items, find_details, summarise_topic, trace_relationship, general_info")
    missing_attributes: list[str]   = Field(default_factory=list, description="Attributes that would help answer better but are not provided")

    def to_context_string(self) -> str:
        parts = [f"Query: {self.raw_query}"]
        if self.domain_hint:          parts.append(f"Domain: {self.domain_hint}")
        if self.topic:                parts.append(f"Topic: {self.topic}")
        if self.entity_focus:         parts.append(f"Entity: {self.entity_focus}")
        if self.context_attributes:   parts.append(f"Context: {', '.join(self.context_attributes)}")
        if self.keywords:             parts.append(f"Keywords: {', '.join(self.keywords)}")
        parts.append(f"Multi-hop: {self.is_multi_hop}  Intent: {self.intent}")
        return " | ".join(parts)


_PLANNER_PROMPT = """\
You are an expert assistant that analyses user queries about information contained in PDF documents.
The documents can cover any subject — science, law, finance, medicine, history, technology, policy, or anything else.

Given the user query below, extract structured information as a JSON object with these keys:
- "domain_hint": the subject domain the query is about (e.g. "climate change", "financial regulation", "public health") — null if completely unclear
- "topic": the specific topic or sub-area within the domain (e.g. "carbon tax mechanisms", "Basel III capital ratios") — null if not clear
- "entity_focus": the primary named entity the user is asking about — a person, organisation, concept, policy, product, law, etc. — null if none
- "context_attributes": list of any contextual filters or constraints mentioned (e.g. "after 2020", "in developing countries", "for small businesses") — empty list if none
- "keywords": list of 3-7 important concepts or terms to search for in documents
- "is_multi_hop": true ONLY if the query explicitly needs information from two clearly separate topic areas that cannot be answered in a single lookup. Prefer false whenever possible.
- "intent": one of "find_information", "explain_concept", "compare_items", "find_details", "summarise_topic", "trace_relationship", "general_info"
- "missing_attributes": list of attributes that would help answer better but are not provided in the query

IMPORTANT: The entire answer will be produced in at most 3 retrieval steps regardless of complexity.
Set is_multi_hop=true only when the query genuinely cannot be answered without retrieving from two distinct topic areas.

Return ONLY valid JSON. No explanation, no markdown fences.


User query: {query}
"""


def _parse_json_response(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from Gemini response:\n{text}")


def plan_query(query: str) -> QueryPlan:
    model = get_gemini_model()
    response = model.generate_content(_PLANNER_PROMPT.format(query=query))
    data = _parse_json_response(response.text.strip())
    data.setdefault("raw_query", query)
    data.setdefault("keywords", [])
    data.setdefault("is_multi_hop", False)
    data.setdefault("intent", "general_info")
    data.setdefault("missing_attributes", [])
    data.setdefault("context_attributes", [])
    plan = QueryPlan(**{k: v for k, v in data.items() if k in QueryPlan.model_fields})
    return plan.model_copy(update={"raw_query": query})


## TESTING ##
# if __name__ == "__main__":
#     import sys

#     BASE_DIR = Path(__file__).resolve().parent.parent
#     if str(BASE_DIR) not in sys.path:
#         sys.path.insert(0, str(BASE_DIR))

#     print(f"\n{'='*60}")
#     print("PLANNER — generic test\n")

#     for q in [
#         "What are the main objectives described in the document?",
#         "Summarise the key findings of the report.",
#         "What evidence is given for the conclusion in chapter 3?",
#     ]:
#         print(f"Query: {q}")
#         try:
#             plan = plan_query(q)
#             print(f"  intent={plan.intent}  multi_hop={plan.is_multi_hop}  keywords={plan.keywords}")
#             print(f"  domain_hint={plan.domain_hint}  topic={plan.topic}  entity_focus={plan.entity_focus}")
#         except Exception as e:
#             print(f"  ERROR: {e}")
#         print()

#     print("Planner test complete.")
