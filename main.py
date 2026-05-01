import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage
from langgraph.graph import START, END, StateGraph
import operator
from langchain_ollama import ChatOllama
from langgraph.types import interrupt, Command

load_dotenv()


model_name = "gemma3:latest"

llm = ChatOllama(
    model=model_name,
    temperature=0.1,
)

config = {"configurable": {"thread_id": "debate-1"}}

class DebateState(TypedDict):
    topic: str
    topic_options: list[str | None] 
    topic_selected: bool       

    round: int
    max_rounds: int
    history: list
    scores: dict
    final_winner: str


initial_state = {
    "topic": "",
    "topic_options": [],
    "topic_selected": False,

    "round": 0,
    "max_rounds": 3,
    "history": [],
    "scores": {"pro": 0, "con": 0},
    "final_winner": ""
}

class Topic(BaseModel):
    topic_title: str = Field(description="The title of the debate topic")
    description: str = Field(description="A brief description of the debate topic")

class TopicList(BaseModel):
    topics: List[Topic] = Field(description="A list of exactly 5 debate topics")


def topic_generator(state: DebateState):
    structured_llm = llm.with_structured_output(TopicList)
    response = structured_llm.invoke("""
        Generate exactly 5 high-quality debate topics.

        Constraints:
        - Each topic must be clear and controversial
        - Max 12 words per topic
        - No duplicates
        - Cover diverse domains (technology, ethics, society, economy)
    """)
    return {
        "topic_options": response.topics,
        "topic_selected": False
    }

def topic_selection(state):
    choice = interrupt({
        "type": "topic_selection",
        "options": state["topic_options"]
    })

    selected_topic = state["topic_options"][choice]

    return {
        **state,
        "topic": selected_topic,
        "topic_selected": True
    }


graph_builder = StateGraph(DebateState)
graph_builder.add_node("topic_generator", topic_generator)
graph_builder.add_node("topic_selection", topic_selection)
graph_builder.add_edge(START, "topic_generator")
graph_builder.add_edge("topic_generator", "topic_selection")
graph_builder.add_edge("topic_selection", END)

graph = graph_builder.compile(config=config)

while True:
    result = graph.invoke(state, config=config)

    # Interrupt happened
    if result.interrupts:
        data = result.interrupts[0].value

        if data["type"] == "topic_selection":
            options = data["options"]

            print("\nChoose a topic:")
            for i, t in enumerate(options):
                print(f"{i}: {t}")

            choice = int(input("Your choice: "))

            # IMPORTANT: replace state with Command
            state = Command(resume=choice)
            continue

    # No interrupt → finished
    break
