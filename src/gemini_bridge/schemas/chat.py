"""Chat and completion request and response schemas."""

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """Request payload for text generation."""

    prompt: str = Field(..., min_length=1, description="User prompt or question.")
    system_instruction: str | None = Field(default=None, description="Optional system prompt or persona.")
    model: str | None = Field(default=None, description="Model override (e.g. gemini-3.6-flash).")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature.")


class ChatCompletionResponse(BaseModel):
    """Response payload for text generation."""

    response: str = Field(..., description="Generated text response from the model.")
    model: str = Field(..., description="Model identifier used for generation.")


class StreamEvent(BaseModel):
    """Single token chunk payload for Server-Sent Events."""

    text: str = Field(..., description="Incremental generated token or text chunk.")
    done: bool = Field(default=False, description="Whether the stream has finished.")
