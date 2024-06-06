from django.db import models
from django.contrib.auth import get_user_model
from .tasks import generate_strings, recognize_string
from .utils import *
from core.middleware import CurrentUserMiddleware


class Page(models.Model):
    class StatusChoices(models.IntegerChoices):
        RECOGNIZED = 0, "Распознано"
        IN_PROGRESS = 1, "В процессе"
        NOT_STARTED = 2, "Не запускалось"

    document = models.ForeignKey("Document", on_delete=models.CASCADE, blank=False, null=False, verbose_name="Документ")
    status = models.IntegerField(choices=StatusChoices.choices, default=StatusChoices.NOT_STARTED, blank=False, null=False, verbose_name="Статус")
    page_num = models.PositiveIntegerField(blank=False, null=False, verbose_name='Номер страницы')
    image = models.ImageField(upload_to=handle_page_img, blank=False, null=False, verbose_name="Изображение страницы")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="Время добавления")
    change_time = models.DateTimeField(auto_now=True, blank=False, null=False, verbose_name="Время последнего изменения")
    last_modified_user = models.ForeignKey(get_user_model(), blank=True, null=True, on_delete=models.SET_NULL, verbose_name="Последний изменявший пользователь", related_name="page_last_modified_user")
    user = models.ForeignKey(get_user_model(), blank=True, null=True, on_delete=models.SET_NULL, verbose_name="Пользователь", related_name="page_user")
    is_demo = models.BooleanField(blank=False, null=False, default=False, verbose_name="Демонстрационный")
    
    class Meta:
        verbose_name_plural = 'Страницы'
        verbose_name = 'Страница'

    def __str__(self) -> str:
         return f"{self.document.name} Страница: {self.page_num}" 
    
    def save(self,  *args, **kwargs):
        user = CurrentUserMiddleware.get_current_user()
        if user and user.is_authenticated:
            self.last_modified_user = user
            if self.user is None:
                self.user = user
        super(Page, self).save(*args, **kwargs)
    
    @property
    def text(self):
        text = "\n"
        strings = String.objects.filter(page=self).order_by('string_num').values_list('text', flat=True)

        return text.join(strings)
    
    def recognize_strings(self):
        self.status = self.StatusChoices.IN_PROGRESS
        self.save()
        #generate_strings(self.pk, self.image.path)
        generate_strings.delay(self.pk, self.image.path)
        
class String(models.Model):
    page = models.ForeignKey(Page, on_delete=models.CASCADE, blank=False, null=False, verbose_name="Страница")
    string_num = models.PositiveIntegerField(blank=False, null=False, verbose_name='Номер строки')
    text = models.TextField(null=False, blank=True, verbose_name="Текст")
    change_time = models.DateTimeField(auto_now=True, verbose_name="Время изменения")
    last_modified_user = models.ForeignKey(get_user_model(), blank=True, null=True, on_delete=models.SET_NULL, verbose_name="Последний изменявший пользователь", related_name="string_last_modified_user")
    user = models.ForeignKey(get_user_model(), blank=True, null=True, on_delete=models.SET_NULL, verbose_name="Пользователь", related_name="string_user")
    is_manual = models.BooleanField(blank=True, null=False, default=False, verbose_name="Изменялось вручную")
    x1 = models.PositiveIntegerField(blank=False, null=True, verbose_name="Координата x1")
    y1 = models.PositiveIntegerField(blank=False, null=True, verbose_name="Координата y1")
    x2 = models.PositiveIntegerField(blank=False, null=True, verbose_name="Координата x2")
    y2 = models.PositiveIntegerField(blank=False, null=True, verbose_name="Координата y2")

    class Meta:
        verbose_name_plural = 'Строки'
        verbose_name = 'Строка'

    def save(self,  *args, **kwargs):
        user = CurrentUserMiddleware.get_current_user()
        if user and user.is_authenticated:
            self.last_modified_user = user
            if self.user is None:
                self.user = user
        super(String, self).save(*args, **kwargs)

    def __str__(self) -> str:
         return f"{self.page.document.name}: {self.page.page_num} страница {self.string_num} строка"
    
    def recognize_text(self):
        #recognize_string(self.pk, self.page.image.path)
        recognize_string.delay(self.pk, self.page.image.path)

    @property
    def coords(self):
        if self.x1 and self.y1 and self.x2 and self.y2:
            return (self.x1, self.y1, self.x2, self.y2)
        else:
            return None

class Document(models.Model):
    class StatusChoices(models.IntegerChoices):
        RECOGNIZED = 0, "Распознано"
        IN_PROGRESS = 1, "В процессе"

    class VisibilityChoices(models.IntegerChoices):
        ALL = 0, "Все"
        MODERATORS = 1, "Модератор и выше"

    last_modified_user = models.ForeignKey(get_user_model(), blank=True, null=True, on_delete=models.SET_NULL, verbose_name="Изменял", related_name="document_last_modified_user")
    user = models.ForeignKey(get_user_model(), blank=True, null=True, on_delete=models.SET_NULL, verbose_name="Создал", related_name="document_user")
    status = models.IntegerField(choices=StatusChoices.choices, blank=False, null=False, default=StatusChoices.IN_PROGRESS, verbose_name="Статус")
    visibility = models.IntegerField(choices=VisibilityChoices.choices, blank=False, null=False, default=VisibilityChoices.ALL, verbose_name="Видимость")
    name = models.CharField(max_length=150, blank=False, null=False, verbose_name="Название")
    description = models.TextField(blank=True, null=False, verbose_name="Описание")
    create_time = models.DateTimeField(auto_now_add=True, blank=False, null=False, verbose_name="Время создания")
    change_time = models.DateTimeField(auto_now=True, blank=False, null=False, verbose_name="Время последнего изменения")
    is_verificated = models.BooleanField(blank=False, null=False, default=False, verbose_name="Верифицировано")
    
    class Meta:
        verbose_name_plural = 'Документы'
        verbose_name = 'Документ'

    def __str__(self) -> str:
         return self.name      
    
    def save(self,  *args, **kwargs):
        user = CurrentUserMiddleware.get_current_user()
        if user and user.is_authenticated:
            self.last_modified_user = user
            if self.user is None:
                self.user = user
        super(Document, self).save(*args, **kwargs)
