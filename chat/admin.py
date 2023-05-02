from django.contrib import admin
from .models import ChatMessage


class ChatMessageAdmin(admin.ModelAdmin):
    pass


# Register your models here.
admin.site.register(ChatMessage, ChatMessageAdmin)