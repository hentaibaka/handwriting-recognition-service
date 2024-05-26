from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import *
from ai_service.models import Metric

    
@receiver(post_save, sender=User)
def update_user_metrics(sender, instance: User, **kwargs):
    metrics = Metric.objects.filter(codename__regex=r'^.*user.*$')
    for metric in metrics: 
        metric.update_metric()
        