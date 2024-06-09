from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.conf import settings
from rest_framework import generics
from .serializers import *


class MetricsView(generics.ListAPIView):
    queryset = Metric.objects.all()
    serializer_class = MetricSerializer 

    @method_decorator(cache_page(settings.CACHE_TTL))
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
    