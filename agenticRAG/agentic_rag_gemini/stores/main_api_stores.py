import asyncio
from typing import Any, Dict, Optional


class InMemoryAnswerJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def set(self, request_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            self._jobs[request_id] = payload

    async def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._jobs.get(request_id)

    async def update(self, request_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._lock:
            job = self._jobs.get(request_id)
            if not job:
                return None
            job.update(updates)
            return job


class InMemoryTaskContextStore:
    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def set(self, task_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            self._tasks[task_id] = payload

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def set_final_answer(self, task_id: str, final_answer: Dict[str, Any]) -> None:
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["final_answer"] = final_answer


answer_job_store = InMemoryAnswerJobStore()
task_context_store = InMemoryTaskContextStore()
