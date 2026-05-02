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

model_name = "gemma3:latest"

llm = ChatOllama(
    model=model_name,
    temperature=0.1,
)

config = {"configurable": {"thread_id": "debate-1"}}

# ------------------ Schemas ------------------

class Topic(BaseModel):
    topic_title: str = Field(description="The title of the debate topic")
    description: str = Field(description="A brief description of the debate topic")

class TopicList(BaseModel):
    topics: List[Topic] = Field(description="A list of exactly 5 debate topics")

# ------------------ State ------------------

class DebateState(TypedDict):
    topic: Dict
    topic_options: List[Dict]
    topic_selected: bool       

    round: int
    max_rounds: int
    history: list
    scores: dict
    final_winner: str

initial_state: DebateState = {
    "topic": {},
    "topic_options": [],
    "topic_selected": False,

    "round": 1,
    "max_rounds": 3,
    "history": [],
    "scores": {"pro": 0, "con": 0},
    "final_winner": ""
}

# ------------------ Nodes ------------------

def topic_generator(state: DebateState):
    structured_llm = llm.with_structured_output(TopicList)

    response = structured_llm.invoke("""
Generate exactly 5 high-quality debate topics.

Constraints:
- Each topic must be clear and controversial
- Max 12 words per topic
- No duplicates
- Cover diverse domains (technology, ethics, society, economy)
- NO HTML
- Plain text only
""")

    return {
        **state,
        "topic_options": [t.dict() for t in response.topics],
        "topic_selected": False
    }


def topic_selection(state: DebateState):
    choice = interrupt({
        "type": "topic_selection",
        "options": state["topic_options"]
    })

    selected_topic = state["topic_options"][choice]

    print(
        f"\nSelected topic: "
        f"{selected_topic['topic_title']} — {selected_topic['description']}"
    )

    return {
        **state,
        "topic": selected_topic,
        "topic_selected": True
    }


def pro_agent(state: DebateState):
    topic = state["topic"]["topic_title"]
    description = state["topic"]["description"]

    history = state["history"]

    prompt = f"""
        You are the PRO side in a debate.

        Topic: {topic}

        Make a strong, clear argument supporting the topic.
        Be concise but persuasive.
        No bullet points, just a paragraph.
    """

    response = llm.invoke(prompt)
    argument = response.content

    return {
        **state,
        'history' : history + [{
            'role': 'pro',
            'content': argument,
            'round' : state['round']
        }]
    }


# ------------------ Graph ------------------

graph_builder = StateGraph(DebateState)

graph_builder.add_node("topic_generator", topic_generator)
graph_builder.add_node("topic_selection", topic_selection)

graph_builder.add_edge(START, "topic_generator")
graph_builder.add_edge("topic_generator", "topic_selection")
graph_builder.add_edge("topic_selection", END)

graph = graph_builder.compile(checkpointer=MemorySaver())

# ------------------ Main Loop ------------------

state = initial_state

while True:
    result = graph.invoke(state, config=config, version="v2")

    # Interrupt handling
    if result.interrupts:
        data = result.interrupts[0].value
        options = data["options"]

        print("\nChoose a topic:\n")
        for i, t in enumerate(options):
            print(f"{i}: {t['topic_title']} — {t['description']}")

        choice = int(input("\nYour choice: "))

        state = Command(resume=choice)
        continue

    # Finished execution
    break
