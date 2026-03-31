from typing import TypedDict

class GraphState(TypedDict):
    query           : str
    context         : str
    source          : str
    is_relevant     : str
    iteration_count : int
    augmented_prompt: str
    answer          : str
    session_id      : str