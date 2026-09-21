"""API v1 root router combining all endpoints."""

from fastapi import APIRouter

from gemini_bridge.api.v1.endpoints.chat import router as chat_router
from gemini_bridge.api.v1.endpoints.rag import router as rag_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(chat_router)
api_router.include_router(rag_router)
