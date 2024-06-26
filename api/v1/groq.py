#!/usr/bin/env python
from pydantic import BaseModel, Field
import httpx
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, Depends, Header, UploadFile, Form, HTTPException
from fastapi.routing import APIRouter
from openai import AsyncClient
import typing

router = APIRouter()


class GroqArgs(BaseModel):
    api_key: str
    model: str
    messages: typing.List[typing.Dict[str, str]]


@router.post("/text")
async def test_groq(args: GroqArgs):
    client = AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        api_key=args.api_key
    )
    return await client.chat.completions.create(
        model=args.model,
        messages=args.messages,
    )

