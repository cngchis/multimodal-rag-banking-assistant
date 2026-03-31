from fastapi import APIRouter, HTTPException
from app.components import ChatRequest, ChatResponse
from src.utils.helper import reset_session
import traceback

router = APIRouter()

# Import graph — lazy init để tránh load model khi import
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        from app.graph import build_graph
        _graph = build_graph()
    return _graph

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        graph = get_graph()
        result = graph.invoke({
            "query": request.query,
            "session_id": request.session_id,
            "iteration_count": 0
        })
        return ChatResponse(
            answer=result["answer"],
            source=result.get("source", ""),
            session_id=request.session_id
        )
    except Exception as e:
        traceback.print_exc()  # ← in full traceback ra terminal
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    reset_session(session_id)
    return {"message": f"Session {session_id} đã được reset"}