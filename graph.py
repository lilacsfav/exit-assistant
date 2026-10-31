import re
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from mdp_solver import get_policy

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
parser = StrOutputParser()

DEMAND_MAP = {"high": "H", "medium": "M", "low": "L"}
GAP_MAP = {"small": "S", "medium": "M", "large": "L"}
DEMAND_NAME = {"H": "High", "M": "Medium", "L": "Low"}
GAP_NAME = {"S": "Small", "M": "Medium", "L": "Large"}
ACTION_NAME = {"Maintain": "Maintain", "PhaseOut": "Phase-Out", "Delete": "Delete"}


class State(TypedDict, total=False):
    question: str
    demand: Optional[str]
    gap: Optional[str]
    action: Optional[str]
    is_valid: bool
    answer: str


# ---------- Parser node (uses LangChain) ----------
parse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract demand and technology gap from the user's message. "
     "Demand must be High, Medium, Low, or None. "
     "Gap must be Small, Medium, Large, or None. "
     "Reply in exactly this format: demand=<value>, gap=<value>"),
    ("human", "{question}"),
])
parse_chain = parse_prompt | llm | parser


def parse_node(state: State):
    text = parse_chain.invoke({"question": state["question"]})
    d = re.search(r"demand\s*=\s*(\w+)", text, re.I)
    g = re.search(r"gap\s*=\s*(\w+)", text, re.I)
    d_val = d.group(1).lower() if d else "none"
    g_val = g.group(1).lower() if g else "none"
    return {"demand": DEMAND_MAP.get(d_val), "gap": GAP_MAP.get(g_val)}


# ---------- Policy node (uses the MDP solver) ----------
def policy_node(state: State):
    action = get_policy(state["demand"], state["gap"])
    return {"action": action}


# ---------- Validator node ----------
def validate_node(state: State):
    ok = state.get("action") in ("Maintain", "PhaseOut", "Delete")
    return {"is_valid": ok}


# ---------- Explainer node (uses LangChain) ----------
explain_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You explain product decisions in simple, plain English. "
     "Use short sentences and easy words. Answer in 2 to 3 sentences."),
    ("human",
     "Demand level: {demand}\n"
     "Technology gap: {gap}\n"
     "Recommended action: {action}\n"
     "Explain this recommendation to a product manager."),
])
explain_chain = explain_prompt | llm | parser


def explain_node(state: State):
    answer = explain_chain.invoke({
        "demand": DEMAND_NAME[state["demand"]],
        "gap": GAP_NAME[state["gap"]],
        "action": ACTION_NAME[state["action"]],
    })
    return {"answer": answer}


# ---------- Clarify node ----------
def clarify_node(state: State):
    return {"answer": (
        "I need two things to help you: the demand level (High, Medium, or Low) "
        "and the technology gap (Small, Medium, or Large). Please include both."
    )}


# ---------- Conditional routers ----------
def route_after_parse(state: State):
    if state.get("demand") and state.get("gap"):
        return "policy"
    return "clarify"


def route_after_policy(state: State):
    return "validate" if state.get("action") else "clarify"


def route_after_validate(state: State):
    return "explain" if state.get("is_valid") else "clarify"


# ---------- Build the graph ----------
builder = StateGraph(State)
builder.add_node("parse", parse_node)
builder.add_node("policy", policy_node)
builder.add_node("validate", validate_node)
builder.add_node("explain", explain_node)
builder.add_node("clarify", clarify_node)

builder.add_edge(START, "parse")
builder.add_conditional_edges("parse", route_after_parse,
                              {"policy": "policy", "clarify": "clarify"})
builder.add_conditional_edges("policy", route_after_policy,
                              {"validate": "validate", "clarify": "clarify"})
builder.add_conditional_edges("validate", route_after_validate,
                              {"explain": "explain", "clarify": "clarify"})
builder.add_edge("explain", END)
builder.add_edge("clarify", END)

graph = builder.compile()
