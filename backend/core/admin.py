from typing import Any
from django.contrib import admin
from django.http import HttpRequest
from .models import User
from django.contrib.auth.models import Group, Permission


#admin.site.unregister(Group)

@admin.register(User)
class AdminUser(admin.ModelAdmin):
    list_display = ('id', 'email', 
                    'first_name', 'last_name', 
                    'middle_name', 'date_joined', 'last_login', 
                    'is_active', 'is_staff',
                    'is_superuser')
    exclude  = ('password', 'user_permissions')#'is_active', 'is_staff', 'is_superuser'
    readonly_fields = ('last_login',)
    list_display_links = ('id',)

    def get_readonly_fields(self, request, obj: User | None=None):
        if obj:
            if request.user.is_superuser:
                return self.readonly_fields
            elif request.user.groups.filter(name='admin').exists():
                return self.readonly_fields + ('is_superuser',)
            else: 
                return self.readonly_fields + ('groups', 'is_active', 'is_staff', 'is_superuser',)
        return self.readonly_fields

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser or request.user.groups.filter(name='admin').exists():
            return qs
        return qs.filter(pk=request.user.pk)

    