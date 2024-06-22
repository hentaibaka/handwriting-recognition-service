from celery import shared_task
from recognition_module.recognition_module import RecognitionModule
from django.apps import apps
import os
from .utils import *


@shared_task()
def generate_strings(page_id, image):
    Page = apps.get_model('documents', 'Page')
    String = apps.get_model('documents', 'String')

    page = Page.objects.get(pk=page_id)
    if page and page.image: # type: ignore

        coords, strings = RecognitionModule.get_lines_and_text(image)

        String.objects.filter(page=page).delete()

        for i, (box, text) in enumerate(zip(coords, strings), start=1):
            string = String.objects.create(page=page, string_num=i,
                                            text=text, is_manual=False,
                                            x1=box[0], y1=box[1],
                                            x2=box[2], y2=box[3])
        
    page.status = page.StatusChoices.RECOGNIZED # type: ignore
    page.save()
        
@shared_task()
def recognize_string(string_id, image):
    String = apps.get_model('documents', 'String')

    string = String.objects.get(pk=string_id)
    if string and string.coords and string.page and string.page.image: # type: ignore
        text = RecognitionModule.get_text_from_line(image, string.coords) # type: ignore
        if text:
            string.text = text # type: ignore
        else:
            string.text = '' # type: ignore
        string.is_manual = False # type: ignore
        string.save()

@shared_task()
def get_pages_from_pdf(doc_id, pdf_file):
    pdf_file_path = f'media/tmp/{pdf_file.name}'
    with open(pdf_file_path, 'wb+') as destination:
        for chunk in pdf_file.chunks():
            destination.write(chunk)
    images = handle_uploaded_pdf(pdf_file_path)
    os.remove(pdf_file_path) 

    Page = apps.get_model('documents', 'Page')
    Document = apps.get_model('documents', 'Document')

    doc = Document.objects.get(pk=doc_id)
    if doc:  
        Page.objects.filter(document=doc).delete()
        for i, image in enumerate(images, start=1):
            Page.objects.create(document=doc,
                                page_num=i,
                                image=image)
