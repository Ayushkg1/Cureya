from rest_framework.response import Response
from rest_framework.views import APIView
from better_profanity import profanity
from rest_framework import permissions


# Create your views here.
class ProfanityView(APIView):
    permission_classes = (permissions.AllowAny, )

    def post(self, request, format=None):
        data = self.request.data
        profanity.load_censor_words()
        output = profanity.censor(data['message'])

        return Response(output)
        
