from rest_framework import generics
from .serializers import *


class MetricsView(generics.ListAPIView):
    queryset = Metric.objects.all()
    serializer_class = MetricSerializer 