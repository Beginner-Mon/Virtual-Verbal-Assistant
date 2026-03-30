from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from schemas.main_api import AnswerRequest, AnswerResponse, QueryRequestCompat, UnifiedTaskResponseCompat
from services.main_api_answer import get_answer_impl, get_answer_status_impl
from services.main_api_payloads import answer_to_query_payload, model_to_dict, query_to_task_payload
from stores.main_api_stores import task_context_store

router = APIRouter(tags=["compatibility"])


@router.post("/query", summary="Compatibility endpoint for AgenticRAG-style clients")
async def query_compat(request: QueryRequestCompat) -> Dict[str, Any]:
    answer = await get_answer_impl(
        AnswerRequest(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_history=request.conversation_history,
            motion_format="glb",
        )
    )
    return answer_to_query_payload(answer, request.query, request.user_id)


@router.post("/process_query", response_model=UnifiedTaskResponseCompat, summary="Submit async query task")
async def process_query_compat(request: QueryRequestCompat) -> UnifiedTaskResponseCompat:
    answer = await get_answer_impl(
        AnswerRequest(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_history=request.conversation_history,
            motion_format="glb",
        )
    )
    task_id = answer.request_id
    await task_context_store.set(
        task_id,
        {
            "query": request.query,
            "user_id": request.user_id,
            "final_answer": model_to_dict(answer) if answer.status == "completed" else None,
        },
    )
    return query_to_task_payload(task_id, answer, request.query, request.user_id)


@router.get("/tasks/{task_id}", response_model=UnifiedTaskResponseCompat, summary="Get async query task status")
async def get_task_compat(task_id: str) -> UnifiedTaskResponseCompat:
    context = await task_context_store.get(task_id)
    if context is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    query = context.get("query", "")
    user_id = context.get("user_id", "default")

    final_answer = context.get("final_answer")
    if final_answer:
        answer_obj = AnswerResponse(**final_answer)
        return query_to_task_payload(task_id, answer_obj, query, user_id)

    try:
        answer_obj = await get_answer_status_impl(task_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        raise

    payload = query_to_task_payload(task_id, answer_obj, query, user_id)
    if payload.status == "completed":
        await task_context_store.set_final_answer(task_id, model_to_dict(answer_obj))

    return payload
