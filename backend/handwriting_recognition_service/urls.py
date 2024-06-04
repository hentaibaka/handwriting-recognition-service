from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from drf_spectacular.views import SpectacularJSONAPIView, SpectacularSwaggerView 
from django.contrib.auth.views import LogoutView
from django.views.generic.base import TemplateView

from django.contrib.auth.views import PasswordResetView
from django.urls import reverse_lazy


class MyPasswordResetView(PasswordResetView):
    email_template_name = 'admin/password_reset_email.html'
    subject_template_name = 'admin/password_reset_subject.txt'
    template_name = 'admin/password_reset.html'
    success_url = reverse_lazy('password_reset_done')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['site_name'] = settings.SITE_NAME
        context['domain'] = settings.EMAIL_DOMAIN
        context['protocol'] = settings.EMAIL_PROTOCOL
        return context

urlpatterns = [
    path('admin/login/', auth_views.LoginView.as_view(template_name='admin/login.html'), name='admin_login'),
    path('admin/password_reset/', MyPasswordResetView.as_view(), name='admin_password_reset'),
    path('admin/password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='admin/password_reset_done.html'), name='password_reset_done'),
    path('admin/reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='admin/password_reset_confirm.html'), name='password_reset_confirm'),
    path('admin/reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='admin/password_reset_complete.html'), name='password_reset_complete'),
    path('admin/', admin.site.urls),
    path('api/', include('documents.urls')),
    path('api/', include('ai_service.urls')),
    path('api/', include('core.urls')),
    path('api/schema/', SpectacularJSONAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='docs'),
    path('', include('social_django.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
