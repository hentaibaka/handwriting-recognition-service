from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.conf import settings
from django.http import HttpResponse

from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import SessionAuthentication
from rest_framework import generics, permissions

from drf_spectacular.utils import extend_schema

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

from PIL import Image
import numpy as np
import cv2
import io

from recognition_module.recognition_module import RecognitionModule
from .serializers import *


class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return None

class DemoDocsView(generics.ListAPIView):
    queryset = Page.objects.all()
    serializer_class = DemoDocsSerializer  

    @method_decorator(cache_page(settings.CACHE_TTL))
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
      

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
        
        image_path = page.image.path
        img = Image.open(image_path)
        img_width, img_height = img.size

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=(img_width, img_height))

        pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))

        c.drawImage(image_path, x=0, y=0)
        c.setFillColorRGB(1, 0, 0)
        c.rotate(180)

        for string in String.objects.filter(page=page):
            x1, y1, x2, y2 = string.x1, string.y1, string.x2, string.y2
            text = string.text
            y1 = img_height - y1  
            y2 = img_height - y2  
            width = x2 - x1
            height = y2 - y1
            text_width = c.stringWidth(text, "Arial", 10)
            font_size = min(width / text_width, height) * 0.5
            c.setFont("Arial", font_size)
            text_width = c.stringWidth(text, "Arial", font_size)
            x_text = x1
            y_text = y2 - 0.25 * height
            c.drawString(-x_text, -y_text, text)
            c.rect(-x1, -y1, -width, -height)
        c.rotate(180)

        c.showPage()
        c.save()

        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{page}.pdf"'

        return response
