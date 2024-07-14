from django.db import models
from django.core.files.storage import FileSystemStorage
import os
from handwriting_recognition_service.settings import BASE_DIR
from django.contrib.auth import get_user_model
from django.apps import apps

from .tasks import *
from django_prometheus.models import ExportModelOperationsMixin


class Metric(ExportModelOperationsMixin("metric"), models.Model):
    name = models.CharField(max_length=32, blank=False, null=False, verbose_name="Название")
    codename = models.CharField(max_length=16, blank=False, null=False, verbose_name="Кодовое название")
    value = models.CharField(max_length=32, blank=False, null=False, verbose_name="Значение")
    
    class Meta:
        verbose_name_plural = 'Метрики'
        verbose_name = 'Метрика'

    def __str__(self) -> str:
         return f"{self.name}: {self.value}"
    
    def update_metric(self):
        if self.codename == "count_doc":
            Document = apps.get_model('documents', 'Document')
            new_value = Document.objects.all().count()
        elif self.codename == "manual_string":
            String = apps.get_model('documents', 'String')
            new_value = String.objects.filter(is_manual=True).count()
        elif self.codename == "auto_string":
            String = apps.get_model('documents', 'String')
            new_value = String.objects.filter(is_manual=False).count()
        elif self.codename == "count_page":
            Page = apps.get_model('documents', 'Page')
            new_value = Page.objects.all().count()
        elif self.codename == "count_user":
            user = get_user_model()
            new_value = user.objects.all().count()
        else:
            return 
        
        self.value = new_value
        self.save()

class AIModel(ExportModelOperationsMixin("aimodel"), models.Model):
    class ModelTypeChoises(models.IntegerChoices):
        EASYOCR = (0, "EasyOCR")
        TROCR = (1, "TrOCR")
        CROCR = (2, "CrOCR")
        DEEPTEXT = (3, "DeepText")
    
    class ModelDetectorChoices(models.IntegerChoices):
        NONE = (0, "None")
        CRAFT = (1, "Craft")
        
    class ModelCorrectorChoices(models.IntegerChoices):
        NONE = (0, "None")
        SAGE = (1, "Sage")
        YT = (2, "YT")
        
    name = models.CharField(max_length=150 , null=False, blank=False, verbose_name="Название модели")
    create_time = models.DateTimeField(auto_now_add=True, blank=False, null=False, verbose_name="Дата создания")
    is_current = models.BooleanField(default=False, blank=False, null=False, verbose_name="Текущая")
    model_type = models.IntegerField(choices=ModelTypeChoises.choices, blank=False, null=False, verbose_name='Тип модели')
    detector = models.IntegerField(choices=ModelDetectorChoices.choices, default=ModelDetectorChoices.CRAFT, blank=False, null=False, verbose_name='Детектор')
    corrector = models.IntegerField(choices=ModelCorrectorChoices.choices, default=ModelCorrectorChoices.NONE, blank=False, null=False, verbose_name='Корректор')

    class Meta:
        verbose_name_plural = 'Модели'
        verbose_name = 'Модель'

    def __str__(self) -> str:
         return f"{self.pk} - {self.name} - {'Текущая' if self.is_current else 'Не используется'} - {self.create_time}"
    
    def set_current(self):
        AIModel.objects.all().update(is_current=False)
        self.is_current = True
        self.save()

class DataSet(ExportModelOperationsMixin("dataset"), models.Model):
    strings = models.ManyToManyField('documents.String', verbose_name="Строки")
    create_time = models.DateTimeField(auto_now_add=True, blank=False, null=False, verbose_name="Дата создания")

    class Meta:
        verbose_name_plural = 'Тренировочные наборы'
        verbose_name = 'Тренировочный набор'

    def __str__(self) -> str:
         return f"{self.pk} - {self.strings.count()} строк - {self.create_time}"
    
class Train(ExportModelOperationsMixin("train"), models.Model):
    class StatusChoices(models.IntegerChoices):
        DONE = 0, "Готово"
        IN_PROGESS = 1, "В процессе"
        NOT_STARTED = 2, "Не запускалось"
        ERROR = 3, "Ошибка"

    train_set = models.ForeignKey(DataSet, on_delete=models.SET_NULL, null=True, blank=False, verbose_name="Тренировочный набор")
    message = models.CharField(max_length=128, default=None, null=True, blank=False, verbose_name="Сводка")
    model_to_train = models.ForeignKey(AIModel, on_delete=models.SET_NULL, null=True, blank=False, verbose_name="Модель для обучения", related_name="models_to_train")
    trained_model = models.ForeignKey(AIModel, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="Обученная модель", related_name="trained_models")
    status = models.IntegerField(choices=StatusChoices.choices, default=StatusChoices.NOT_STARTED, null=False, blank=False, verbose_name="Статус")
    create_time = models.DateTimeField(auto_now_add=True, blank=False, null=False, verbose_name="Дата создания")
    user = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, blank=False, null=True, verbose_name="Пользователь")
    num_iter = models.PositiveIntegerField(default=100, blank=False, null=False, verbose_name="Кол-во итераций")
    val_interval = models.PositiveIntegerField(default=50, blank=False, null=False, verbose_name="Валидационный интервал")
    batch_size = models.PositiveBigIntegerField(default=64, blank=False, null=False, verbose_name="Размер партии")

    class Meta:
        verbose_name_plural = 'Обучения'
        verbose_name = 'Обучение'

    def __str__(self) -> str:
         return f"{self.pk} - {self.create_time}"

    def start_train(self):
        self.status = self.StatusChoices.IN_PROGESS
        self.save()
        
        #sstart_train(self.pk)
        start_train.delay(self.pk)
 
    @property
    def dataset_log(self) -> str:
        if self.trained_model and self.model_to_train:
            log_file = os.path.join(RecognitionModule.TRAINS_PATH, f'model-{self.model_to_train.pk}-dataset-{self.pk}', 'log_dataset.txt')
            if os.path.isfile(log_file):
                with open(log_file, 'r', encoding='utf-8') as file:
                    content = file.read()
                    return content
        return '-'
    
    @property
    def train_log(self) -> str:
        if self.trained_model and self.model_to_train:
            log_file = os.path.join(RecognitionModule.TRAINS_PATH, f'model-{self.model_to_train.pk}-dataset-{self.pk}', 'log_train.txt')
            if os.path.isfile(log_file):
                with open(log_file, 'r', encoding='utf-8') as file:
                    content = file.read()
                    return content
        return '-'

    @property
    def opt(self) -> str:
        if self.trained_model and self.model_to_train:
            log_file = os.path.join(RecognitionModule.TRAINS_PATH, f'model-{self.model_to_train.pk}-dataset-{self.pk}', 'opt.txt')
            if os.path.isfile(log_file):
                with open(log_file, 'r', encoding='utf-8') as file:
                    content = file.read()
                    return content
        return '-'
    