from django.db import models


class ChatMessage(models.Model):
    message = models.TextField(blank=True, null=True)
    response = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
