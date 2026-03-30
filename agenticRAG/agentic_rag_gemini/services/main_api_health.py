from typing import Dict
import asyncio

import httpx


async def check_services_health(agentic_rag_url: str, dart_url: str) -> Dict[str, str]:
    """Ping downstream services and return a status map."""
    statuses: Dict[str, str] = {}

    async def check_service(name: str, base_url: str) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{base_url}/health")
                statuses[name] = "ok" if response.status_code == 200 else f"http_{response.status_code}"
            except Exception as exc:
                statuses[name] = f"unreachable ({type(exc).__name__})"

    await asyncio.gather(
        check_service("agenticrag", agentic_rag_url),
        check_service("dart", dart_url),
    )
    return statuses
