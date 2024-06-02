from collections.abc import Sequence
from django.contrib import admin
from django.core.exceptions import ValidationError
from django.contrib.auth.models import Group
from django.http import HttpRequest
from django.utils.html import format_html
from django.shortcuts import redirect
from django.urls import reverse, path
from .models import *
from .forms import *


@admin.register(Metric)
class AdminMetric(admin.ModelAdmin):
    list_display = ('id', 'name', 'codename', 'value')
    list_display_links = ('id',)
    ordering = ('id',)
    readonly_fields = ('codename', 'value')
    list_editable = ('name', )

    def save_model(self, request, obj, form, change):
        instance: Metric = form.save(commit=False)
        
        instance.update_metric()

        instance.save()
        form.save_m2m()
        return instance

@admin.register(AIModel)
class AdminAIModel(admin.ModelAdmin):
    list_display = ('id', 'name', 'model_type', 'create_time', 'is_current', 'make_model_current_button')
    list_display_links = ('id', 'name')
    readonly_fields = ('name', 'is_current', 'model_type')
    ordering = ('create_time',)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                '<pk>/make_current/',
                self.admin_site.admin_view(self.make_model_current_action),
                name='aimodel_make_current',
            ),
        ]
        return custom_urls + urls
    
    def make_model_current_action(self, request, pk):
        obj = self.get_object(request, pk)
        obj.set_current()

        self.message_user(request, f"{obj.name} теперь используется")
        return redirect(request.META['HTTP_REFERER'])
    

    def make_model_current_button(self, obj):
        if obj.is_current:
            return format_html('<button class="button" disabled>Текущая</button>')
        
        return format_html('<a class="button" href="{}">Сделать текущей</a>',
                           reverse('admin:aimodel_make_current', args=[obj.pk]))
    
    make_model_current_button.short_description = 'Сделать текущей'
    make_model_current_button.allow_tags = True

@admin.register(DataSet)
class AdminDataSet(admin.ModelAdmin):
    form = DataSetForm
    list_display = ('id', 'create_time')
    list_display_links = ('id',)
    ordering = ('-create_time',)

@admin.register(Train)
class AdminTrain(admin.ModelAdmin):
    list_display = ('id', 'train_set', 'model_to_train', 'trained_model', 'status', 'message', 'create_time', 'user',)
    list_display_links = ('id',)
    readonly_fields = ('dataset_log', 'train_log', 'opt', 'message', 'user', 'trained_model', 'status')
    ordering = ('-create_time',)
    search_fields = ['user__first_name', 'user__last_name', 'user__patronymic']
    list_filter = ['user', 'status', 'create_time']

    def get_readonly_fields(self, request, obj: Train | None=None):
        if obj and obj.status in [obj.StatusChoices.DONE, obj.StatusChoices.IN_PROGESS]:
            return self.readonly_fields + ('train_set', 'model_to_train', 'num_iter', 'val_interval', 'batch_size')
        return self.readonly_fields
    
    def get_list_display(self, request):
        if self.has_start_train_button_permission(request.user):
            return self.list_display + ('start_train_button', )
        else: 
            return self.list_display

    def has_start_train_button_permission(self, user):
        return user.groups.filter(name__in=['moderator', 'admin']).exists() or user.is_superuser

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('<pk>/start_train/', self.admin_site.admin_view(self.start_train_action), name='start_train_action'),
        ]
        return custom_urls + urls
    
    def start_train_action(self, request, pk):
        if self.has_start_train_button_permission(request.user):
            obj = self.get_object(request, pk)
            obj.start_train() # type: ignore

            self.message_user(request, f"Обучение {obj} запущено")
            return redirect(request.META['HTTP_REFERER'])
        else:
            return redirect('admin:ai_service_train_changelist')
    
    def start_train_button(self, obj: Train):
        if obj.status in [obj.StatusChoices.NOT_STARTED, obj.StatusChoices.ERROR]:
            return format_html('<a class="button" href="{}">Начать обучение</a>',
                               reverse('admin:start_train_action', args=[obj.pk]))
        elif obj.status == obj.StatusChoices.IN_PROGESS:
            return format_html('<button class="button" disabled>В процессе обучения</button>')
        else:
            return format_html('<button class="button" disabled>Обучено</button>')
    
    start_train_button.short_description = 'Обучение'
    start_train_button.allow_tags = True

    def save_model(self, request, obj, form, change): 
        if obj.num_iter < obj.val_interval:
            raise ValidationError("Кол-во итераций не должно быть меньше валидационного интервала")

        instance = form.save(commit=False)
        if instance.user is None:
            instance.user = request.user
        instance.save()
        form.save_m2m()
        return instance
    