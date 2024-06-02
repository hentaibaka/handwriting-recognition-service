from collections.abc import Sequence
from django.contrib import admin
from django.contrib.admin.views.main import ChangeList
from django.contrib.auth import get_user_model
from django.http import HttpRequest
from django.utils.safestring import mark_safe
from django.urls import path, reverse
from django.shortcuts import render, redirect
import os
from .models import *
from .forms import *


User = get_user_model()

class PageInline(admin.TabularInline):
    model = Page
    extra = 1
    #classes = ('collapse', )
    can_delete = True
    show_change_link = True

class StringInline(admin.StackedInline):
    model = String
    extra = 1
    #classes = ('collapse', )
    can_delete = True
    show_change_link = True

@admin.register(Document)
class AdminDocument(admin.ModelAdmin):
    list_display = ('id', 'user', 'name', 'create_time', 'status', 'visibility', 'is_verificated')
    list_display_links = ('id', 'name')
    ordering = ('-create_time', 'name')
    search_fields = ('name', 'user__first_name', 'user__last_name', 'user__patronymic')
    list_filter = ('user', 'status', 'visibility', 'is_verificated', 'create_time')
    readonly_fields = ('user', )
    inlines = (PageInline, )
    actions = ('set_visibility_to_one', 'set_visibility_to_zero', 'set_verificated', 'set_not_verificated')

    def get_actions(self, request):
        actions = super().get_actions(request)
        if not request.user.groups.filter(name__in=['admin', 'moderator']).exists() and not request.user.is_superuser:
            if 'set_visibility_to_zero' in actions:
                del actions['set_visibility_to_zero']
            if 'set_visibility_to_one' in actions:
                del actions['set_visibility_to_one']
            if 'set_verificated' in actions:
                del actions['set_verificated']
            if 'set_not_verificated' in actions:
                del actions['set_not_verificated']
        return actions

    @admin.action(description='Скрыть от библиотекарей')
    def set_visibility_to_one(self, request, queryset):
        queryset.update(visibility=1)
        self.message_user(request, f"{len(queryset)} документов скрыто от библиотекарей")

    @admin.action(description='Сделать видимыми для всех')
    def set_visibility_to_zero(self, request, queryset):
        queryset.update(visibility=0)
        self.message_user(request, f"{len(queryset)} документов сделано видимыми для всех")

    @admin.action(description='Пометить как верифицированные')
    def set_verificated(self, request, queryset):
        queryset.update(verificated=True)
        self.message_user(request, f"{len(queryset)} документов помечено как верифицированные")

    @admin.action(description='Пометить как не верифицированные')
    def set_not_verificated(self, request, queryset):
        queryset.update(verificated=False)
        self.message_user(request, f"{len(queryset)} документов помечено как не верифицированные")

    def get_readonly_fields(self, request, obj=None):
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return self.readonly_fields
        return self.readonly_fields + ('visibility', 'is_verificated')
    
    def get_list_editable(self, request):
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return ('status', 'visibility', 'is_verificated',)
        return ('status',)

    def changelist_view(self, request, extra_context=None):
        self.list_editable = self.get_list_editable(request)
        return super(AdminDocument, self).changelist_view(request, extra_context)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return qs
        return qs.filter(visibility=0)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                '<int:document_id>/change/upload_pdf/',
                self.admin_site.admin_view(self.upload_pdf),
                name='document-upload-pdf',
            ),
        ]
        return custom_urls + urls
    
    def upload_pdf(self, request, document_id):
        if request.method == "POST":
            form = PDFUploadForm(request.POST, request.FILES)
            if form.is_valid():
                pdf_file = form.cleaned_data['pdf_file']
                # Сохранение файла временно
                pdf_file_path = f'media/tmp/{pdf_file.name}'
                with open(pdf_file_path, 'wb+') as destination:
                    for chunk in pdf_file.chunks():
                        destination.write(chunk)
                # Конвертация страниц PDF в изображения
                images = handle_uploaded_pdf(pdf_file_path)
                os.remove(pdf_file_path)  # Удаление временного файла после обработки

                print(images)

                # Обработка изображений, например, сохранение в модель
                # your_model_instance.images = images
                # your_model_instance.save()
                return redirect('admin:documents_document_change', document_id)
        else:
            form = PDFUploadForm()
        context = dict(
            self.admin_site.each_context(request),
            form=form,
            document_id=document_id,
        )
        return render(request, 'admin/upload_pdf.html', context)

    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = extra_context or {}
        extra_context['upload_pdf_url'] = f'upload_pdf/'
        return super(AdminDocument, self).change_view(request, object_id, form_url, extra_context=extra_context)


    def save_model(self, request, obj, form, change): 
        instance = form.save(commit=False)
        if instance.user is None:
            instance.user = request.user
        instance.save()
        form.save_m2m()
        return instance

@admin.register(Page)
class AdminPage(admin.ModelAdmin):
    list_display = ('id', 'document', 'page_num', 'create_time', 'status', 'is_demo')
    list_display_links = ('id',)
    readonly_fields = ('text',)
    ordering = ('document', 'page_num')
    search_fields = ('document__name',)    
    list_filter = ('document__name', 'status', 'create_time', 'is_demo')
    inlines = (StringInline, )
    save_on_top = True
    form = PageForm
    actions = ('set_is_demo_true', 'set_is_demo_false', 'recognize_queryset')

    def get_actions(self, request):
        actions = super().get_actions(request)
        if not request.user.groups.filter(name__in=['admin', 'moderator']).exists() and not request.user.is_superuser:
            if 'set_is_demo_true' in actions:
                del actions['set_is_demo_true']
            if 'set_is_demo_false' in actions:
                del actions['set_is_demo_false']
        return actions

    @admin.action(description='Пометить как демонстрационные')
    def set_is_demo_true(self, request, queryset):
        queryset.update(is_demo=True)
        self.message_user(request, f"{len(queryset)} страниц помечено как демонстрационные")
    
    @admin.action(description='Пометить как не демонстрационные')
    def set_is_demo_false(self, request, queryset):
        queryset.update(is_demo=False)
        self.message_user(request, f"{len(queryset)} страниц помечено как не демонстрационные")

    @admin.action(description='Автоматически распознать текст')
    def recognize_queryset(self, request, queryset):
        for page in queryset:
            page.recognize_strings()
        self.message_user(request, f"Отправлен запрос на распознавание {len(queryset)} страниц")

    def get_readonly_fields(self, request, obj=None):
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return self.readonly_fields
        return self.readonly_fields + ('is_demo',)
    
    def get_list_editable(self, request):
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return ('is_demo', 'status',)
        return ('status',)

    def changelist_view(self, request, extra_context=None):
        self.list_editable = self.get_list_editable(request)
        if extra_context is None:
            extra_context = {}
        extra_context['actions_with_confirmation'] = ['recognize_queryset']
        return super(AdminPage, self).changelist_view(request, extra_context=extra_context)        

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return qs
        return qs.filter(document__visibility=0)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                '<pk>/recognize_strings/',
                self.admin_site.admin_view(self.recognize_strings_view),
                name='recognize_strings',
            ),
        ]
        return custom_urls + urls

    def recognize_strings_view(self, request, pk):
        obj = self.get_object(request, pk)
        obj.recognize_strings()

        self.message_user(request, "Запрос на распознавание строк отправлен")
        return redirect(request.META['HTTP_REFERER'])

    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = extra_context or {}
        extra_context['recognize_strings'] = reverse('admin:recognize_strings', args=[object_id])
        return super().change_view(request, object_id, form_url, extra_context=extra_context)

    def save_model(self, request, obj, form, change):
        instance = form.save(commit=False)
        
        #print("page save")

        instance.save()
        form.save_m2m()
        return instance

@admin.register(String)
class AdminString(admin.ModelAdmin):
    list_display = ('id', 'page', 'string_num', 'text', 'change_time', 'is_manual')
    list_display_links = ('id',)
    ordering = ('page', 'string_num')
    search_fields = ('page__document__name', 'text')
    list_filter = ('page__document__name', 'is_manual', 'change_time')
    list_editable = ('is_manual', )
    actions = ('set_is_manual_true', 'set_is_manual_false', 'recognize_queryset')
    save_on_top = True
    save_as = True
    save_as_continue = True
    form = StringForm

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return qs
        return qs.filter(page__document__visibility=0)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                '<pk>/recognize_text/',
                self.admin_site.admin_view(self.recognize_text_view),
                name='recognize_text',
            ),
        ]
        return custom_urls + urls

    def recognize_text_view(self, request, pk):
        obj = self.get_object(request, pk)
        obj.recognize_text()

        self.message_user(request, "Запрос на распознавание текста отправлен")
        return redirect(request.META['HTTP_REFERER'])

    def image_tag(self, obj):
        width = obj.x2 - obj.x1
        height = obj.y2 - obj.y1
        return f'''
        <div style="width: {width}px; height: {height}px; overflow: hidden; position: relative;">
            <img src="{obj.page.image.url}" style="position: absolute; left: -{obj.x1}px; top: -{obj.y1}px;" />
        </div>
        '''

    image_tag.short_description = 'Cropped Image'
    image_tag.allow_tags = True

    def changelist_view(self, request, extra_context=None):
        if extra_context is None:
            extra_context = {}
        extra_context['actions_with_confirmation'] = ['recognize_queryset']
        return super(AdminString, self).changelist_view(request, extra_context=extra_context)

    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = extra_context or {}
        extra_context['image_tag'] = self.image_tag(self.get_object(request, object_id))
        extra_context['recognize_text'] = reverse('admin:recognize_text', args=[object_id])
        return super().change_view(request, object_id, form_url, extra_context=extra_context)

    def render_change_form(self, request, context, *args, **kwargs):
        if 'image_tag' in context:
            context['adminform'].form.fields['image_preview'].widget.attrs['style'] = 'display: none;'
        return super().render_change_form(request, context, *args, **kwargs)

    def get_readonly_fields(self, request, obj=None):
        if obj:
            return self.readonly_fields + ('image_preview')
        return self.readonly_fields

    @admin.action(description='Пометить как распознанные вручную')
    def set_is_manual_true(self, request, queryset):
        queryset.update(is_manual=True)
        self.message_user(request, f"{len(queryset)} строк помечено как распознанные вручную")

    @admin.action(description='Пометить как распознанные автоматически')
    def set_is_manual_false(self, request, queryset):
        queryset.update(is_manual=False)
        self.message_user(request, f"{len(queryset)} строк помечено как распознанные автоматически")

    @admin.action(description='Автоматически распознать текст')
    def recognize_queryset(self, request, queryset):
        for string in queryset:
            string.recognize_text()
        self.message_user(request, f"Отправлен запрос на распознавание {len(queryset)} строк")


admin.site.site_header = "Сервис распознавания рукописного текста"
admin.site.index_title = "Рабочее место сотрудника"
