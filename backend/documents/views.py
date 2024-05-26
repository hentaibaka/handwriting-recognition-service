from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from drf_spectacular.utils import extend_schema
import numpy as np
import cv2

from recognition_module.recognition_module import RecognitionModule
from .serializers import *


class DemoDocsView(generics.ListAPIView):
    queryset = Page.objects.all()
    serializer_class = DemoDocsSerializer    

class RecognizeImage(APIView):
    parser_classes = [MultiPartParser, ]

    serializer_class = RecognizeImageSerializer

    @extend_schema(responses=str)
    def post(self, request, format=None):
        serializer = RecognizeImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        text = ""
        image = request.FILES['image']
        if image:
            image = cv2.imdecode(np.frombuffer(image.read() , np.uint8), cv2.IMREAD_UNCHANGED)
            strings = RecognitionModule.get_lines_and_text(image)
            text = " ".join([string[1] for string in strings]).replace('\n', '')      

        response_serializer = RecognizeTextSerializer(data={"text": text})
        response_serializer.is_valid(raise_exception=True)

        return Response(response_serializer.validated_data)