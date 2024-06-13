from pdf2image import convert_from_path
from django.core.files.base import ContentFile
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from PIL import Image
import io
from django.apps import apps
from django.core.files.uploadedfile import InMemoryUploadedFile

def handle_uploaded_pdf(pdf_file_path, quality=100):
    # Конвертация страниц PDF в изображения
    images = convert_from_path(pdf_file_path, dpi=96)
    image_files = []
    
    for i, image in enumerate(images):
        img_byte_arr = io.BytesIO()
        rgb_image = image.convert('RGB') 
        rgb_image.save(img_byte_arr, format='JPEG', quality=quality)
        img_byte_arr = img_byte_arr.getvalue()
        image_files.append(ContentFile(img_byte_arr, name=f'page_{i + 1}.jpg'))
    
    return image_files


def handle_page_img(instance, filename: str) -> str:
    new_filename = f"{instance.document.name}_{instance.page_num}.jpg"
    upload_path = f"images/{instance.document.name}/{new_filename}"

    # Конвертируем изображение в JPG
    image = Image.open(instance.image)
    rgb_image = image.convert('RGB')  # Конвертируем изображение в RGB, если оно не в этом формате
    jpg_image_io = io.BytesIO()
    rgb_image.save(jpg_image_io, format='JPEG')
    jpg_image_io.seek(0)

    # Создаем новый InMemoryUploadedFile для сохранения
    instance.file = InMemoryUploadedFile(
        jpg_image_io,
        None,
        new_filename,
        'image/jpeg',
        jpg_image_io.getbuffer().nbytes,
        None
    )

    return upload_path

def generate_pdf_doc(pages):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
    String = apps.get_model('documents', 'String')
    for page in pages:
        image = page.image.path
        strings = String.objects.filter(page=page)
        if image:
            generate_pdf_page(c, image, strings)
    
    c.save()
    buffer.seek(0)
    return buffer

def generate_pdf_page(canvas, image, strings):
    img = Image.open(image)
    img_width, img_height = img.size
    canvas.setPageSize((img_width, img_height))  
    
    canvas.drawImage(image, x=0, y=0, width=img_width, height=img_height)
    canvas.setFillColorRGB(1, 0, 0)
    canvas.rotate(180)

    for string in strings:
        x1, y1, x2, y2 = string.x1, string.y1, string.x2, string.y2
        text = string.text

        y1 = img_height - y1  
        y2 = img_height - y2  

        width = x2 - x1
        height = y2 - y1

        text_width = canvas.stringWidth(text, "Arial", 10)
        font_size = min(width / text_width, height) * 0.5
        canvas.setFont("Arial", font_size)
        text_width = canvas.stringWidth(text, "Arial", font_size)

        x_text = x1
        y_text = y2 - 0.25 * height

        canvas.drawString(-x_text, -y_text, text)
        canvas.rect(-x1, -y1, -width, -height)

    canvas.rotate(180)
    canvas.showPage()
    