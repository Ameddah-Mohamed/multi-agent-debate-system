import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from langchain_ollama import ChatOllama

load_dotenv()

# ------------------ LLM ------------------

llm = ChatOllama(
    model="gemma3:latest",
    temperature=0.1,
)

config = {"configurable": {"thread_id": "debate-1"}}

# ------------------ Schemas ------------------

class Topic(BaseModel):
    topic_title: str
    description: str

class TopicList(BaseModel):
    topics: List[Topic]

class Score(BaseModel):
    pro_agent_score: int
    pro_score_reason: str
    con_agent_score: int
    con_score_reason: str

# ------------------ State ------------------

class DebateState(TypedDict):
    topic: Dict
    topic_options: List[Dict]
    topic_selected: bool

    round: int
    max_rounds: int
    history: List[Dict]

    scores: Dict
    scores_reason: Dict
    final_winner: str

initial_state: DebateState = {
    "topic": {},
    "topic_options": [],
    "topic_selected": False,
    "round": 1,
    "max_rounds": 3,
    "history": [],
    "scores": {"pro": 0, "con": 0},
    "scores_reason": {},
    "final_winner": ""
}

# ------------------ Nodes ------------------

def topic_generator_agent(state: DebateState):
    structured_llm = llm.with_structured_output(TopicList)

    response = structured_llm.invoke("""
Generate exactly 5 high-quality debate topics.

Constraints:
- Clear and controversial
- Max 12 words
- No duplicates
- Diverse domains
- Plain text only
""")

    return {
        **state,
        "topic_options": [t.model_dump() for t in response.topics],
        "topic_selected": False
    }


def topic_selection(state: DebateState):
    choice = interrupt({
        "type": "topic_selection",
        "options": state["topic_options"]
    })

    selected_topic = state["topic_options"][choice]

    print(
        f"\nSelected topic:\n"
        f"{selected_topic['topic_title']} — {selected_topic['description']}\n"
    )

    return {
        **state,
        "topic": selected_topic,
        "topic_selected": True
    }


def pro_agent(state: DebateState):
    topic = state["topic"]["topic_title"]
    history = state["history"]

    prompt = f"""
You are the PRO side in a debate.

Topic: {topic}

Previous debate:
{history}

Make a NEW strong argument supporting the topic.
Do NOT repeat previous arguments.
Be persuasive and concise.
"""

    response = llm.invoke(prompt)
    argument = response.content

    print("\n[PRO]:\n", argument)

    return {
        **state,
        "history": history + [{
            "role": "pro",
            "content": argument,
            "round": state["round"]
        }]
    }


def con_agent(state: DebateState):
    topic = state["topic"]["topic_title"]
    last_argument = state["history"][-1]["content"]

    prompt = f"""
You are the CON side in a debate.

Topic: {topic}

Opponent argument:
{last_argument}

Refute it and argue against the topic.
Be sharp, logical, and persuasive.
Do NOT repeat earlier arguments.
"""

    response = llm.invoke(prompt)
    counter_argument = response.content

    print("\n[CON]:\n", counter_argument)

    return {
        **state,
        "history": state["history"] + [{
            "role": "con",
            "content": counter_argument,
            "round": state["round"]
        }],
        "round": state["round"] + 1
    }


def judge_agent(state: DebateState):
    topic = state["topic"]["topic_title"]
    history = state["history"]

    debate_text = ""
    for h in history:
        debate_text += f"{h['role'].upper()}: {h['content']}\n\n"

    prompt = f"""
You are an expert debate judge.

Topic: {topic}

Debate transcript:
{debate_text}

Task:
- Score both sides (0–10)
- Explain reasoning
- Decide winner
"""

    structured_llm = llm.with_structured_output(Score)
    output = structured_llm.invoke(prompt)

    print("\n[JUDGE DECISION]")
    print(output)

    winner = "pro" if output.pro_agent_score > output.con_agent_score else "con"

    return {
        **state,
        "final_winner": winner,
        "scores": {
            "pro": output.pro_agent_score,
            "con": output.con_agent_score
        },
        "scores_reason": {
            "pro": output.pro_score_reason,
            "con": output.con_score_reason
        }
    }

# ------------------ Graph ------------------

graph_builder = StateGraph(DebateState)

graph_builder.add_node("topic_generator", topic_generator_agent)
graph_builder.add_node("topic_selection", topic_selection)
graph_builder.add_node("pro_agent", pro_agent)
graph_builder.add_node("con_agent", con_agent)
graph_builder.add_node("judge_agent", judge_agent)

graph_builder.add_edge(START, "topic_generator")
graph_builder.add_edge("topic_generator", "topic_selection")
graph_builder.add_edge("topic_selection", "pro_agent")
graph_builder.add_edge("pro_agent", "con_agent")

def should_continue(state: DebateState):
    if state["round"] > state["max_rounds"]:
        return "judge"
    return "continue"

graph_builder.add_conditional_edges(
    "con_agent",
    should_continue,
    {
        "judge": "judge_agent",
        "continue": "pro_agent"
    }
)

graph_builder.add_edge("judge_agent", END)

graph = graph_builder.compile(checkpointer=MemorySaver())

# ------------------ Main Loop ------------------

state = initial_state

while True:
    result = graph.invoke(state, config=config, version="v2")

    if result.interrupts:
        data = result.interrupts[0].value
        options = data["options"]

        print("\nChoose a topic:\n")
        for i, t in enumerate(options):
            print(f"{i}: {t['topic_title']} — {t['description']}")

        choice = int(input("\nYour choice: "))
        state = Command(resume=choice)
        continue

    break

# ------------------ Final Output ------------------

print("\n=== FINAL RESULT ===\n")

print("Winner:", result.value["final_winner"])
print("\nScores:", result.value["scores"])
print("\nReasoning:", result.value["scores_reason"])