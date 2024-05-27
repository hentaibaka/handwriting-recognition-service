from cProfile import label
from fileinput import filename
from tabnanny import verbose
from rest_framework import serializers
from documents.models import Page
from documents.models import *
from PIL import Image
import magic


class DemoDocsSerializer(serializers.ModelSerializer):
    url = serializers.SerializerMethodField('get_image_url')
    weight = serializers.SerializerMethodField('get_image_weight')
    width = serializers.SerializerMethodField('get_image_width')
    height = serializers.SerializerMethodField('get_image_height')
    type = serializers.SerializerMethodField('get_image_type')
    text = serializers.SerializerMethodField('get_text_on_image')
    name = serializers.SerializerMethodField('get_demodoc_name')

    def get_image_weight(self, obj: Page) -> int:
        return obj.image.size
    
    def get_image_width(self, obj: Page) -> int:
        return obj.image.width
    
    def get_image_height(self, obj: Page) -> int:
        return obj.image.height
    
    def get_image_type(self, obj: Page) -> str:
        print(obj.image.path)
        mime_type = magic.from_file(obj.image.path, mime=True)
        return mime_type

    def get_image_url(self, obj: Page) -> str:
        return obj.image.url
    
    def get_text_on_image(self, obj: Page) -> str:
        strings = String.objects.filter(page=obj).order_by('string_num').values_list('text', flat=True)
        if strings:
            return "".join(strings)
        else:
            return ""
        
    def get_demodoc_name(self, obj: Page) -> str:
        return str(obj)

    class Meta:
        model = Page
        fields = ('url', 'weight', 'width', 'height', 'type', 'text', 'name')

class RecognizeImageSerializer(serializers.Serializer):
    image = serializers.ImageField(use_url=False)

class RecognizeTextSerializer(serializers.Serializer):
    text = serializers.CharField(allow_blank=True)
        
