from celery import shared_task
from recognition_module.recognition_module import RecognitionModule
from django.apps import apps


@shared_task()
def generate_strings(page_id, image):
    Page = apps.get_model('documents', 'Page')
    String = apps.get_model('documents', 'String')

    page = Page.objects.get(pk=page_id)

    String.objects.filter(page=page).delete()

    if page and page.image: # type: ignore

        strings = RecognitionModule.get_lines_and_text(image)

        for i, (box, text) in enumerate(strings, start=1):
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
        string.text = text # type: ignore
        string.is_manual = False # type: ignore
        string.save()
        