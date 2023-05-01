from rest_framework.views import APIView
from rest_framework.parsers import JSONParser

from braces.views import CsrfExemptMixin

from django.conf import settings

from .utils import *

import openai
import pandas as pd
from scipy import spatial


ERR_MESSAGE_NO_QUESTION_PROVIDED = "Bạn chưa cung cấp câu hỏi!"
ERR_UNKNOWN = "Đã xảy ra lỗi!" 


class ChatAnswerView(CsrfExemptMixin, APIView):
    parser_classes = [JSONParser]
    authentication_classes = []


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        openai.api_key = settings.OPENAI_API_KEY
        self.embedding_model = settings.EMBEDDING_MODEL
        self.gpt_model = settings.GPT_MODEL
        self.df = pd.read_csv(settings.EMBEDDING_PATHS)
        self.prompt_introduction = settings.PROMPT_INTRODUCTION


    # search function
    def strings_ranked_by_relatedness(
        self,
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.Embedding.create(
            model=self.embedding_model,
            input=query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]


    # synthesize GPT prompt
    def query_message(
        self,
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = self.strings_ranked_by_relatedness(query, df)
        introduction = self.prompt_introduction
        question = f"\n\Câu hỏi cần trả lời: {query}"
        message = introduction
        for string in strings:
            next_article = f'\n\Thông tin:\n"""\n{string}\n"""'
            if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
            ):
                break
            else:
                message += next_article
        return message + question


    # answer function
    def ask(
        self,
        query: str,
        df: pd.DataFrame,
        model: str = GPT_MODEL,
        token_budget: int = 2048,
        print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = self.query_message(query, df, model=model, token_budget=token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": settings.CHATBOT_LABEL},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message


    # API
    def post(self, request, *args, **kwargs):
        question = request.data.get('question')

        if question is None:
            return error_response(1, ERR_MESSAGE_NO_QUESTION_PROVIDED)
        
        try:
            answer = self.ask(question, self.df)
        except Exception as e:
            return error_response(2, ERR_UNKNOWN)
        
        return ai_response(answer)