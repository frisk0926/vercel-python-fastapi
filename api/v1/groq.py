#!/usr/bin/env python
import typing
import pydantic
from fastapi import Header, APIRouter
from openai import AsyncClient

router = APIRouter()

class ChatArgs(pydantic.BaseModel):
    model: str
    messages: typing.List[typing.Dict[str, str]]

@router.post("/lovegroq")
async def groq_api(args: ChatArgs, authorization: str = Header(...)):
    api_key = authorization.split(" ")[1]
    client = AsyncClient(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    return await client.chat.completions.create(
        model=args.model,
        messages=args.messages,
    )
