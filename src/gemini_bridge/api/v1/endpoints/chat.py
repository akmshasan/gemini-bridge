"""Chat and completion endpoints."""

import json
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from gemini_bridge.core.config import Settings, get_settings
from gemini_bridge.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from gemini_bridge.services.gemini import GeminiService

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_gemini_service(settings: Annotated[Settings, Depends(get_settings)]) -> GeminiService:
    """Dependency provider for GeminiService."""
    try:
        return GeminiService(settings=settings)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(err),
        ) from err


@router.post("/generate")
async def generate_chat(
    request: ChatCompletionRequest,
    service: Annotated[GeminiService, Depends(get_gemini_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatCompletionResponse:
    """Generate a full non-streaming completion for a prompt."""
    try:
        text = await service.generate_text(
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            model=request.model,
            temperature=request.temperature,
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {err}",
        ) from err
    else:
        return ChatCompletionResponse(
            response=text,
            model=request.model or settings.gemini_model,
        )


@router.post("/stream")
async def stream_chat(
    request: ChatCompletionRequest,
    service: Annotated[GeminiService, Depends(get_gemini_service)],
) -> StreamingResponse:
    """Stream completion tokens in real-time via Server-Sent Events (SSE)."""

    async def event_generator() -> AsyncGenerator[str]:
        try:
            async for chunk in service.generate_stream(
                prompt=request.prompt,
                system_instruction=request.system_instruction,
                model=request.model,
                temperature=request.temperature,
            ):
                payload = json.dumps({"text": chunk, "done": False})
                yield f"data: {payload}\n\n"
            final_payload = json.dumps({"text": "", "done": True})
            yield f"data: {final_payload}\n\n"
        except Exception as err:  # noqa: BLE001
            err_payload = json.dumps({"error": str(err), "done": True})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
