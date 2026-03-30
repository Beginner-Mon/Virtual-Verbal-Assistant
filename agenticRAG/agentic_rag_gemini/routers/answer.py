from fastapi import APIRouter

from schemas.main_api import AnswerRequest, AnswerResponse
from services.main_api_answer import get_answer_impl, get_answer_status_impl

router = APIRouter(tags=["answer"])


@router.post("/answer", response_model=AnswerResponse, summary="Get text answer + motion + speech")
async def get_answer(request: AnswerRequest) -> AnswerResponse:
    return await get_answer_impl(request)


@router.get("/answer/status/{request_id}", response_model=AnswerResponse, summary="Get async enrichment status")
async def get_answer_status(request_id: str) -> AnswerResponse:
    return await get_answer_status_impl(request_id)
