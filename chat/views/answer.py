from rest_framework.views import APIView
from rest_framework.parsers import JSONParser

from braces.views import CsrfExemptMixin

from django.conf import settings

import openai
import pandas as pd
from scipy import spatial

class ChatAnswerView(CsrfExemptMixin, APIView):
    parser_classes = [JSONParser]
    authentication_classes = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        openai.api_key = settings.OPENAI_API_KEY
        self.embedding_model = settings.EMBEDDING_MODEL
        self.gpt_model = settings.GPT_MODEL
        self.df = pd.read_csv(settings.EMBEDDING_PATHS)

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
