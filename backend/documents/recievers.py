from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import *
from ai_service.models import Metric

    
@receiver(post_save, sender=String)
def update_string_metrics(sender, instance: String, **kwargs):
    metrics = Metric.objects.filter(codename__regex=r'^.*string.*$')
    for metric in metrics: 
        metric.update_metric()

@receiver(post_save, sender=Page)
def update_page_metrics(sender, instance: Page, **kwargs):
    metrics = Metric.objects.filter(codename__regex=r'^.*page.*$')
    for metric in metrics: 
        metric.update_metric()

@receiver(post_save, sender=Document)
def update_doc_metrics(sender, instance: Document, **kwargs):
    metrics = Metric.objects.filter(codename__regex=r'^.*doc.*$')
    for metric in metrics: 
        metric.update_metric()