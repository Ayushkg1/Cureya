from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import permissions
from .emotion_logic import emotion_teller

class EmotionView(APIView):
    permission_classes = (permissions.AllowAny, )

    def post(self, request, format=None):
        data = self.request.data
        output = emotion_teller(data['message'])
        return Response(output)
        
