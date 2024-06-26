#!/usr/bin/env python
import typing
from fastapi import File, UploadFile, Header, APIRouter, Depends
from pydantic import BaseModel
from openai import AsyncClient
from fastapi.responses import JSONResponse
from fastapi import HTTPException

router = APIRouter()

class WhisperArgs(BaseModel):
    model: str
    prompt: typing.Optional[str] = None
    response_format: typing.Optional[str] = "json"
    language: typing.Optional[str] = "en"
    temperature: typing.Optional[float] = 0.0

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    args: WhisperArgs = Depends(WhisperArgs),
    authorization: str = Header(...)
):
    api_key = authorization.split(" ")[1]
    client = AsyncClient(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    contents = await file.read()
    
    try:
        transcription = await client.audio.transcriptions.create(
            file=(file.filename, contents),
            model=args.model,
            prompt=args.prompt,
            response_format=args.response_format,
            language=args.language,
            temperature=args.temperature
        )
        return JSONResponse(content={"transcription": transcription.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
