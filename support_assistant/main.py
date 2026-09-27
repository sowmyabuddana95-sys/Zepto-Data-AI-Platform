import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# MOCK_LLM is the required deterministic baseline.
# MOCK_LLM=1 -> deterministic mock mode
# MOCK_LLM=0 -> optional real LLM generation
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"


# ============================================================
# EMBEDDING MODEL + CHROMADB
# ============================================================

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def load_documents():
    """Load all policy documents from docs/."""

    documents = []
    ids = []
    metadatas = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        documents.append(text)
        ids.append(path.stem)
        metadatas.append(
            {
                "doc_id": path.stem,
                "filename": path.name,
            }
        )

    return documents, ids, metadatas


def build_index():
    """Embed all documents and store them in ChromaDB."""

    existing_count = collection.count()

    if existing_count > 0:
        return

    documents, ids, metadatas = load_documents()

    if len(documents) != 8:
        raise RuntimeError(
            f"Expected 8 policy documents, but found {len(documents)}."
        )

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# Build the index when the application starts.
build_index()


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_context: list[dict]
    response: dict


# ============================================================
# PROMPT SKELETON
# ============================================================

STRUCTURED_PROMPT = """
ROLE:
You are a Zepto customer-support policy assistant.

CONTEXT:
Answer only from the retrieved Zepto policy context supplied below.

TASK:
Answer the user's question using the retrieved context.

FORMAT:
Return valid JSON with exactly these fields:
{
  "answer": "string",
  "sources": ["doc_id"],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise and directly relevant.

NEGATIVE CONSTRAINT:
Do not invent policies, prices, timings, refunds, or procedures that are not supported by the retrieved context.
If the context does not support the answer, say that the available policy context does not provide that information.

FEW-SHOT EXAMPLE:
User: "How long can I return a damaged grocery item?"
Context: "Groceries and perishables can be returned within 24 hours if damaged, spoiled, or incorrect."
Output:
{
  "answer": "Damaged groceries can be returned within 24 hours.",
  "sources": ["doc_02"],
  "confidence": 0.95
}

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# NODE 1: CLASSIFY INTENT
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state: GraphState) -> GraphState:
    """
    Deterministic mock classifier.

    A query is a policy_question if it contains any required
    Zepto policy keyword.
    """

    query = state["query"].lower()

    is_policy = any(keyword in query for keyword in POLICY_KEYWORDS)

    intent = "policy_question" if is_policy else "general_question"

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_top_three(query: str):
    """Retrieve exactly the top 3 documents using cosine similarity."""

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    retrieved = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        retrieved.append(
            {
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i] if i < len(distances) else None,
            }
        )

    return retrieved


# ============================================================
# OPTIONAL REAL LLM SUPPORT
# ============================================================

def real_llm_generate(prompt: str):
    """
    Optional Groq-based generation.

    This is NOT required for the graded baseline.
    MOCK_LLM=1 remains the default.
    """

    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant",
        ),
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content


def generate_validated_response(
    prompt: str,
    fallback_response: AskResponse,
) -> AskResponse:
    """
    Optional real-LLM path.

    Retries up to 2 additional times when the returned JSON
    fails Pydantic validation.
    """

    last_error = None

    for attempt in range(3):
        try:
            raw = real_llm_generate(prompt)

            if isinstance(raw, str):
                data = json.loads(raw)
            else:
                data = raw

            return AskResponse(**data)

        except Exception as exc:
            last_error = exc

            prompt = prompt + f"""

The previous response failed schema validation:
{last_error}

Retry and return ONLY valid JSON with:
answer, sources, confidence.
"""

    # If all retries fail, return a deterministic safe response.
    return fallback_response


# ============================================================
# NODE 2: RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(state: GraphState) -> GraphState:

    query = state["query"]

    retrieved = retrieve_top_three(query)

    if not retrieved:
        fallback = AskResponse(
            answer="No relevant Zepto policy context was retrieved.",
            sources=[],
            confidence=0.0,
        )

        return {
            **state,
            "retrieved_context": [],
            "response": fallback.model_dump(),
        }

    # Required deterministic mock answer.
    top_chunk = retrieved[0]["document"]
    top_snippet = top_chunk[:200]

    mock_response = AskResponse(
        answer=f"Based on the retrieved context: {top_snippet}",
        sources=[item["id"] for item in retrieved],
        confidence=1.0,
    )

    if MOCK_LLM:
        return {
            **state,
            "retrieved_context": retrieved,
            "response": mock_response.model_dump(),
        }

    # Optional real LLM branch.
    context = "\n\n".join(
        f"[{item['id']}] {item['document']}"
        for item in retrieved
    )

    prompt = STRUCTURED_PROMPT.format(
        query=query,
        context=context,
    )

    response = generate_validated_response(
        prompt,
        mock_response,
    )

    return {
        **state,
        "retrieved_context": retrieved,
        "response": response.model_dump(),
    }


# ============================================================
# NODE 3: DIRECT ANSWER
# ============================================================

def direct_answer(state: GraphState) -> GraphState:

    mock_response = AskResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0,
    )

    if MOCK_LLM:
        return {
            **state,
            "retrieved_context": [],
            "response": mock_response.model_dump(),
        }

    prompt = STRUCTURED_PROMPT.format(
        query=state["query"],
        context="No policy retrieval was performed because this is a general question.",
    )

    response = generate_validated_response(
        prompt,
        mock_response,
    )

    return {
        **state,
        "retrieved_context": [],
        "response": response.model_dump(),
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_classification(state: GraphState):
    if state["intent"] == "policy_question":
        return "policy_question"

    return "general_question"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(GraphState)

graph_builder.add_node(
    "classify_intent",
    classify_intent,
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph_builder.add_node(
    "direct_answer",
    direct_answer,
)

graph_builder.add_edge(
    START,
    "classify_intent",
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END,
)

graph_builder.add_edge(
    "direct_answer",
    END,
)

graph = graph_builder.compile()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="Policy-grounded Zepto support assistant using LangGraph and ChromaDB.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "mock_llm": MOCK_LLM,
        "status": "running",
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):

    result = graph.invoke(
        {
            "query": request.query,
        }
    )

    return AskResponse(
        **result["response"]
    )