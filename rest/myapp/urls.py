from django.urls import path
from .views import openai_example, link_upload, chat

urlpatterns = [
    path("", link_upload, name="link_upload"),
    path("open/", openai_example, name="openai_example"),
    path("chat/", chat, name="chat"),
]
