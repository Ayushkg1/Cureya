from django.urls import path
from .views import ProfanityView

urlpatterns = [
    path('',ProfanityView.as_view()),
]