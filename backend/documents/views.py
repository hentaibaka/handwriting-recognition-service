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
import requests

from recognition_module.recognition_module import RecognitionModule
from recognition_module.recognize_function import correct_text
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

    @staticmethod
    def upload_photo(file_bytes: io.BytesIO) -> dict:
        url = "https://rehand.ru/api/v1/upload"
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Host": "rehand.ru",
            "Origin": "https://rehand.ru",
            "Referer": "https://rehand.ru/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/52.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\""
        }
        
        files = {
            'file': ('image.jpg', file_bytes, 'image/jpg')
        }
        
        data = {
            "language": "russian",
            "correctOrder": "on",
            "speller": "on",
            "handwritingText": "on"
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()  # Raises an HTTPError for bad responses
        except requests.RequestException as e:
            return {"error": "Request failed.", "exception": str(e)}
    
        if response.status_code == 200 and response.text:
            try:
                result = response.json()
                return result
            except ValueError:
                return {"error": "Response is not JSON."}
        else:
            return {"error": "Request failed.", "status_code": response.status_code, "response_text": response.text}

    @extend_schema(responses=RecognizeTextSerializer)
    def post(self, request, format=None):
        serializer = RecognizeImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        text = "Текст не найден"
        image = request.FILES['image'].read()

        bytes = io.BytesIO(image)

        json_response = self.upload_photo(bytes)
        if "error" in json_response:
            print(json_response.get('error'))
            if image:
                image = cv2.imdecode(np.frombuffer(image , np.uint8), cv2.IMREAD_UNCHANGED)
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

                    strings = RecognitionModule.get_lines_and_text(image)
                    if strings:
                        text = "<br>".join([string[1] for string in strings])   
                elif image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    strings = RecognitionModule.get_lines_and_text(image)
                    if strings:
                        text = "<br>".join([string[1] for string in strings])   
                else:
                    pass  
        else:
            output_text = json_response.get('output_text', '')
            output_text = correct_text(output_text)
            text = output_text
          
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
