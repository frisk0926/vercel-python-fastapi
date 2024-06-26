#!/usr/bin/env python
import typing
from fastapi import File, UploadFile, Header, APIRouter, Depends, HTTPException
from pydantic import BaseModel
from openai import AsyncClient
from fastapi.responses import JSONResponse
from fastapi import Form

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
    model: str = Form(...),
    prompt: typing.Optional[str] = Form(None),
    response_format: typing.Optional[str] = Form("json"),
    language: typing.Optional[str] = Form("en"),
    temperature: typing.Optional[float] = Form(0.0),
    authorization: str = Header(...)
):
    api_key = authorization.split(" ")[1]
    client = AsyncClient(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    contents = await file.read()
    
    try:
        transcription = await client.audio.transcriptions.create(
            file=(file.filename, contents),
            model=model,
            prompt=prompt,
            response_format=response_format,
            language=language,
            temperature=temperature
        )
        return JSONResponse(content={"transcription": transcription.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
