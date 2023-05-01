from django.urls import path

from .views.answer import ChatAnswerView


urlpatterns = [
    # Chat Answering APIs
    path("answer/", ChatAnswerView.as_view(), name="answer"),
]