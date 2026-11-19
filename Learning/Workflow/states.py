""" 
        We are creating a Graph
        the first thing needed is STATE


"""

import os

#1) to create a state we used typed DICT

from typing import TypedDict

class State(TypedDict):
    topic : str
    summary : str
    score : int

#2) 2nd way using pydantic
#   It is good at data validationn and type checking at validation

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic : str
    summary : str
    score : int = ""

    @field_validator
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("score must be positive")


#3 ) python data classes
# Standard Python data class but it is used rarely

from dataclasses import dataclass, field

@dataclass
class State:
    topic : str =""
    summary : str= ""
    messages : str = field(default_factory=list)


#4 ) Using langgraph

from langgraph.graph import MessagesState

class State(MessagesState):
    #messages field is already included with add_messages reducer
    # just add your extra fields

    username : str
    language : str