import openai
import tiktoken

from django.conf import settings

from rest_framework.response import Response
from rest_framework import status


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def num_tokens(text: str, model: str = settings.EMBEDDING_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def error_response(error_code, description):
    result = {
        "error_code": error_code,
        "ai_response": description
    }
    return Response(result, status=status.HTTP_200_OK)


def ai_response(string):
    result = {
        "error_code": 0,
        "ai_response": string
    }
    return Response(result, status=status.HTTP_200_OK)