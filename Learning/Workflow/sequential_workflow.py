import os
from typing import TypedDict


#state

class pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

# NOdes Creation
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7)

#for node creation we need python fn

def editor_node(state : pipelinestate) -> dict:
    """
        Stage 1 : Cleans up grammar , removes typos ,and refines the tone
    """
    prompt = (
        "You are an expert copyeditor. Clean up the following raw text."
        "Fix any grammatical errors, spelling mistakes and smooth out the transition flow"
        "While keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    return {"edited_text" : response.content.strip()}

#ScriptWriter NOde

def scriptwriter_node(state: pipelinestate) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""
    print("\n --- [Stage 2]: Executing Script Writer NOde ---")

    prompt=(
        "You are a charismatic Youtube content creator.Take this edited text and transform"
        "it into a highly engaging ,punchy ,conversational video script hook. Make it sounds"
        "like a real person speaking passionately, Return only the script content.\n\n"
        f"Edited Text {state['edited_text']}"
    )
    response = llm.invoke(prompt)
    return {
        "script_text":response.content.strip()
    }

def translator_node(state : pipelinestate) -> dict:
    """Stage 3 : Translates the script into natural flowing Hinglish."""
    print("\n---[STAGE 3]: Executing Hinglish Translator Node ---")
    prompt = (
        "You are an expert content localizer for the Indian Market. Take the following script"
        "and convert it into natural,flowing 'Hinglish'. Do not simply translate it sentence "
        "or repeat information. Altering comfortably between Hindi and English phrases"
        "an intellectual tech educator would speak naturally in a live stram.Keep the energy high"
        "Return only the final Hinglish text\n\n"
        f"Script:{state['script_text']}"
    )

    response = llm.invoke(prompt)
    return {"final_output":response.content.strip()}

#Connect these nodes(Edges)
#As state and nodes are ready it is time to create the Graph
#and for creating the graph you have to connect these nodes and for that you have to use edes

# edges are bery important to create the workflows

from langgraph.graph import StateGraph, START ,END


#create the graph
graph =StateGraph(pipelinestate)

#add the nodes
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

#sequential workflow
#add edges one after other

graph.add_edge(START,"editor")
graph.add_edge("editor","scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)

#Compile the Graph

app = graph.compile()

result=app.invoke({
    "raw_input" : "Ai agents are the future of texh. They can think ,plan ,and act on their .Langgraph helps you build agents with proper control and memory."
    })
#print output
print("Your results are ====")
print(result['final_output'])

#It is a scriptwriter taking the raw input,then edit,scriptwrite,translate then it connent edges using langgraph "