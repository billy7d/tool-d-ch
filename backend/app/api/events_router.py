import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.services.translation.worker import translation_worker

router = APIRouter(prefix="/api/events", tags=["Live Events"])


@router.get("")
async def stream_events(request: Request):
    queue = asyncio.Queue()

    def listener(event_data):
        try:
            queue.put_nowait(event_data)
        except Exception:
            pass

    translation_worker.register_event_listener(listener)

    async def event_generator():
        try:
            # Initial ping event
            yield f"data: {json.dumps({'type': 'CONNECTED', 'message': 'SSE Connected'})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield f"data: {json.dumps({'type': 'PING'})}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
