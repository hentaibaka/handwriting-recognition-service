from collections.abc import Sequence
from django.contrib import admin
from django.contrib.admin.views.main import ChangeList
from django.contrib.auth import get_user_model
from django.http import HttpRequest
from django.utils.safestring import mark_safe
from django.urls import path, reverse
from django.shortcuts import redirect

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

    def get_readonly_fields(self, request, obj=None):
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return self.readonly_fields
        return self.readonly_fields + ('visibility', 'is_verificated')
    
    def get_changelist_instance(self, request: HttpRequest) -> ChangeList:
        cl = super().get_changelist_instance(request)
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            cl.list_editable = ('status', 'visibility', 'is_verificated')
        else:
            cl.list_editable = ('status',)
        return cl

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return qs
        return qs.filter(visibility=0)

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

    def get_readonly_fields(self, request, obj=None):
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            return self.readonly_fields
        return self.readonly_fields + ('is_demo',)
    
    def get_changelist_instance(self, request: HttpRequest) -> ChangeList:
        cl = super().get_changelist_instance(request)
        if request.user.groups.filter(name__in=['moderator', 'admin']).exists() or request.user.is_superuser:
            cl.list_editable = ('is_demo', 'status',)
        else:
            cl.list_editable = ('status',)
        return cl

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
    actions = ('set_is_manual_true', 'set_is_manual_false')
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

    @admin.action(description='Пометить как распознанные автоматически')
    def set_is_manual_false(self, request, queryset):
        queryset.update(is_manual=False)


admin.site.site_header = "Сервис распознавания рукописного текста"
admin.site.index_title = "Рабочее место сотрудника"
