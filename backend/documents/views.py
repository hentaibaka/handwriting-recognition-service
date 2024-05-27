from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from drf_spectacular.utils import extend_schema
import numpy as np
import cv2

from recognition_module.recognition_module import RecognitionModule
from .serializers import *
from rest_framework.authentication import SessionAuthentication
from rest_framework import permissions

class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return None

class DemoDocsView(generics.ListAPIView):
    queryset = Page.objects.all()
    serializer_class = DemoDocsSerializer    

class RecognizeImage(APIView):
    parser_classes = [MultiPartParser, ]
    permission_classes = (permissions.AllowAny, )
    authentication_classes = (CsrfExemptSessionAuthentication,)
    serializer_class = RecognizeImageSerializer

    @extend_schema(responses=str)
    def post(self, request, format=None):
        serializer = RecognizeImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        text = "Текст не найден"
        image = request.FILES['image']
        if image:
            image = cv2.imdecode(np.frombuffer(image.read() , np.uint8), cv2.IMREAD_UNCHANGED)
            print(image.shape)
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

                strings = RecognitionModule.get_lines_and_text(image)
                if strings:
                    text = "\n".join([string[1] for string in strings])   
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                strings = RecognitionModule.get_lines_and_text(image)
                if strings:
                    text = "\n".join([string[1] for string in strings])   
            else:
                pass
              

        response_serializer = RecognizeTextSerializer(data={"text": text})
        response_serializer.is_valid(raise_exception=True)

        return Response(response_serializer.validated_data)