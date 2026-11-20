#PROJECT IDEA
"""
    1. Toxicity Monitor
    2. The Copyright COrp
    3. Rhe Cultural NOde
    """

from dotenv import load_dotenv
load_dotenv()
import os
from typing import TypedDict,Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.1)


def merge_score_dicts(existing : dict, newupdate : dict) -> dict:
    if existing is None:
        return newupdate
    return{**existing, **newupdate}


class AnalyzerState(TypedDict):
    raw_text : str
    #we need to use 3 types here  the raw string going to 3 nodes and ending at one node safety_score(/|\)
    #when after completing one node it store score in sfaety_score then if the second one done it rewrites the old one so we loose data
    #we need reducer (merge_score_dicts)
    safety_score : Annotated[dict[str , int],merge_score_dicts]

# Creating nodes

def toxicity_node(state: AnalyzerState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under our single state key
    return {"safety_score":{"toxity_level":score}}

def copyright_node(state: AnalyzerState) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response=llm.invoke(prompt)

    try:
        score=int(response.content.strip())
    except ValueError:
        score = 0
    return {"safety_score":{"copyright_risk":score}}

def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score":{"cultural_intensity":score}}


builder = StateGraph(AnalyzerState)

builder.add_node("toxicity",toxicity_node)
builder.add_node("copyright",copyright_node)
builder.add_node("cultural",culture_node)

builder.add_edge(START,"toxicity")
builder.add_edge(START,"copyright")
builder.add_edge(START,"cultural")

builder.add_edge("toxicity",END)
builder.add_edge("copyright",END)
builder.add_edge("cultural",END)


app=builder.compile()
sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """

    
initial_state = {
    "raw_text": sample_script,
    "safety_score": {} # Initialized as an empty dictionary
}

final_state = app.invoke(initial_state)

print(final_state["safety_score"])