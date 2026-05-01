import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

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



