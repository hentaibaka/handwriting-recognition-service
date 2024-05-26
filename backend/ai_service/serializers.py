from pyexpat import model
from attr import field
from rest_framework import serializers
from .models import *


class MetricSerializer(serializers.ModelSerializer):
    class Meta: 
        model = Metric
        fields = ('name', 'value')