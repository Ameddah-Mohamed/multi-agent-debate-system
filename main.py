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

load_dotenv()


model_name = "gemma3:latest"

llm = ChatOllama(
    model=model_name,
    temperature=0.1,
)

class DebateState(TypedDict):
    topic: str
    topic_options: list[str]  
    topic_selected: bool       

    round: int
    max_rounds: int
    history: list
    scores: dict
    final_winner: str



class TopicList(BaseModel):
    topics: List[str] = Field(
        description="A list of exactly 5 debate topics"
    )


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