from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from drf_spectacular.utils import extend_schema
from django.http import HttpResponse
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from PIL import Image
import cv2
import io

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

    @extend_schema(responses=RecognizeTextSerializer)
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

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Добавление изображения
        image_path = page.image.path
        img = Image.open(image_path)
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        c.drawImage(image_path, 0, height - (width * aspect), width=width, height=width * aspect)

        # Добавление ограничивающих рамок и текста
        for string in String.objects.filter(page=page):
            c.setStrokeColorRGB(1, 0, 0)  # Красный цвет для рамок
            c.setLineWidth(2)
            c.rect(string.x1, height - string.y2, string.x2 - string.x1, string.y2 - string.y1)  # Рисование рамки
            c.drawString(string.x1 + 2, height - string.y1 - 12, string.text)  # Добавление текста

        c.showPage()
        c.save()

        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{page}.pdf"'

        return response
