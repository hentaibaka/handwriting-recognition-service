from pdf2image import convert_from_path
from django.core.files.base import ContentFile
from PIL import Image
import io

def handle_uploaded_pdf(pdf_file_path):
    # Конвертация страниц PDF в изображения
    images = convert_from_path(pdf_file_path)
    image_files = []
    for i, image in enumerate(images):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        image_files.append(ContentFile(img_byte_arr, name=f'page_{i + 1}.png'))
    
    return image_files


def handle_page_img(instance, filename: str) -> str:
    filename = f"{instance.document.name}_{instance.page_num}.{filename.split('.')[1]}"
    return f"images/{instance.document.name}/{filename}"
