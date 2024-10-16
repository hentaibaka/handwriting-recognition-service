import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from drf_spectacular.utils import extend_schema
from django.http import HttpResponse
import numpy as np

import cv2
import io

from recognition_module.recognition_module import RecognitionModule
from .serializers import *
from .utils import generate_pdf_doc
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

    @extend_schema(responses=RecognizeTextSerializer)
    def post(self, request, format=None):
        serializer = RecognizeImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        text = "Текст не найден"
        image = request.FILES['image'].read()

        if image:
            image = cv2.imdecode(np.frombuffer(image , np.uint8), cv2.IMREAD_UNCHANGED)

            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

                _, strings = RecognitionModule.get_lines_and_text(image)

                if strings:
                    text = "<br>".join(strings) 

            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                _, strings = RecognitionModule.get_lines_and_text(image)

                if strings:
                    text = "<br>".join(strings)
          
        response_serializer = RecognizeTextSerializer(data={"text": text})
        response_serializer.is_valid(raise_exception=True)

        return Response(response_serializer.validated_data)
    
class PageTextFileView(APIView):
    def get(self, request, page_id, *args, **kwargs):
        try:
            page = Page.objects.get(pk=page_id)
        except Page.DoesNotExist:
            return HttpResponse(status=404)

        # Получение содержимого поля text
        generated_text = page.text

        # Создание файла в памяти
        file_buffer = io.BytesIO()
        file_buffer.write(generated_text.encode('utf-8'))
        file_buffer.seek(0)

        # Подготовка ответа с файлом
        response = HttpResponse(file_buffer, content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename={page}.txt'

        return response

class PagePDFFileView(APIView):
    def get(self, request, page_id, *args, **kwargs):
        try:
            page = Page.objects.get(pk=page_id)
        except Page.DoesNotExist:
            return HttpResponse(status=404)
        
        buffer = generate_pdf_doc((page,))

        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{page}.pdf"'

        return response
