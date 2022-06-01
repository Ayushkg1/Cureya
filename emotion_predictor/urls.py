from django.urls import path
from .views import EmotionView

urlpatterns = [
    path('',EmotionView.as_view()),
]