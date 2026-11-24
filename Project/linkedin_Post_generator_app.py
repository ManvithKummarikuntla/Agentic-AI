r"""
Using 2 loops
writer -. tools
reviewer -> writer
tools + TOolNode

LINKEDIN POST GENERATOR

-> Topic
    |
<                                                  ###ITERATIVE WORKFLOW
N1 Generator <------|
    |               |
N2 Reviewer         |FEEDBACK           ## FEEDBACK LOOP OR REACT LOOP
    |               |
    /\              |
  YES NO-------------
    |
    |
    |
   POST

   # we are going to use 2 diff llms for single task
   # MULTI AGENT POST GENERATOR
        """

import os
import streamlit as st
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv("C:/Users/my pc/Desktop/Agentic_AI/Learning/Langgraph/Workflow/.env")
#llm_google=

search_tool = TavilySearch(max_results = 3)

tools=[search_tool]

writer_llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash", temperature = 0.3)

#state building
class State(TypedDict):
    topic : str
    messages : Annotated[list,add_messages]
    draft : str
    review_feedback : str
    is_approved : bool
    attempt : int

# gemini returns content as a list of blocks, groq returns a string
def get_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)

# creation of nodes

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150-200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

def writer_node (state : State) -> dict:
    """Writer (or rewrites) the LinkedIn post. Can call Tavily to search first."""

    # coming back from the tools node: continue the same attempt with the search results
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        response = writer_llm_with_tools.invoke(
            [("system",WRITER_SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages" : [response]}

    attempt = state.get("attempt",0) + 1
    topic = state["topic"]
    previous_feedback = state["review_feedback"]

    if attempt ==1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}. "
            f"If you need current info search the web first"
        )
    else:
        user_message =(
            f"Your previous draft on '{topic}' was rejected. "
            f"Here is the reviewer's feedback \n\n {previous_feedback} \n\n"
            f"Write a new, improved draft that fixes every issue mentioned. "
            f"Do not repeat the same mistake"
        )
    human = HumanMessage(content=user_message)
    messages = [("system",WRITER_SYSTEM_PROMPT), *state["messages"], human]

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages" : [human, response],
        "attempt" : attempt

    }

tool_node = ToolNode(tools)

def extract_draft_node(state: State) ->dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state["messages"][-1]
    draft= get_text(last_message.content)
    print(f"\n\nGenerated Post \n {draft}\n")

    return {"draft": draft}

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)

def reviewer_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state['draft']

    prompt=(
        f"review the LinkedIn post draft: \n"
        f"{draft} \n"
        f"give the review"
    )

    response =reviewer_llm.invoke([("system",REVIEWER_SYSTEM_PROMPT),("human",prompt)])
    review_text = get_text(response.content).strip()

    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]
    # ["this will be part of review("APPROVED OR REJECTED")","this will be part of FEEDBACK"]
    # so we are going to find out approved using upper of and we are searching by splitting at feedback section and searhcing the 1st part
    # as mentioned [0]

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:",1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[VERDICT: {verdict}]")
    print(f"[FEEDBACK: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved" : is_approved
    }

# Router Function

def should_use_tool(state: State):
    last_message = state["messages"][-1]

    if getattr(last_message,"tool_calls",None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state: State):
    if state["is_approved"]:
        print("POST has been Approved \n")
        return END
    if state['attempt']>=3:
        print("Reached MAX attempts")
        return END
    return "writer"

#build the Graph

graph = StateGraph(State)
graph.add_node("writer",writer_node)
graph.add_node("tools",tool_node)
graph.add_node("extract_draft",extract_draft_node)
graph.add_node("reviewer",reviewer_node)

graph.add_edge(START,"writer")

graph.add_conditional_edges(
    "writer", should_use_tool
)
graph.add_edge("tools","writer")
graph.add_edge("extract_draft" , "reviewer")

graph.add_conditional_edges(
    "reviewer", should_stop_looping
)

app = graph.compile()



# ---------------------------- STREAMLIT UI ----------------------------

st.set_page_config(page_title="LinkedIn Post Generator", page_icon="📝", layout="centered")

st.title("📝 LinkedIn Post Generator")
st.caption("A writer agent drafts the post (and can search the web). A reviewer agent checks it and sends it back until it's publish-ready.")

topic = st.text_input("What topic do you want a LinkedIn post about?", placeholder="e.g. Is deep learning required for AI engineer roles?")
generate = st.button("Generate post", type="primary")

NODE_LABELS = {
    "writer": "✍️ Writer is drafting...",
    "tools": "🔎 Searching the web...",
    "extract_draft": "📄 Draft ready",
    "reviewer": "🧐 Reviewer is checking...",
}

if generate:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        initial_state = {
            "topic": topic.strip(),
            "messages": [],
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
        }

        final_state = dict(initial_state)

        try:
            with st.status("Starting generation...", expanded=True) as status:
                for step in app.stream(initial_state, stream_mode="updates"):
                    for node, update in step.items():
                        if not update:
                            continue
                        # keep the latest values (messages not needed for display)
                        for k, v in update.items():
                            if k != "messages":
                                final_state[k] = v

                        status.update(label=NODE_LABELS.get(node, node))

                        if node == "extract_draft":
                            st.markdown(f"**Draft (attempt {final_state['attempt']})**")
                            st.info(update["draft"])
                        elif node == "reviewer":
                            if update["is_approved"]:
                                st.success(f"APPROVED — {update['review_feedback']}")
                            else:
                                st.error(f"REJECTED — {update['review_feedback']}")

                status.update(label="Done", state="complete", expanded=False)

            st.divider()
            st.subheader("Final LinkedIn post")
            st.text_area("Copy your post", final_state["draft"], height=320)

            c1, c2 = st.columns(2)
            c1.metric("Total attempts", final_state["attempt"])
            c2.metric("Approved", "Yes" if final_state["is_approved"] else "No (max attempts)")

        except Exception as e:
            st.error(f"Something went wrong: {e}")