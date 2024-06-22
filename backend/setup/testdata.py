# -*- coding: utf-8 -*-
from ai_service.models import *
from documents.models import *
from django.contrib.auth import get_user_model

User = get_user_model()

model = AIModel.objects.create(name="best_old",
                               is_current=False, 
                               model_type=0)
model = AIModel.objects.create(name="best_new",
                               is_current=False, 
                               model_type=0)
model = AIModel.objects.create(name="trocr",
                               is_current=False, 
                               model_type=1)
model = AIModel.objects.create(name="crocr",
                               is_current=True, 
                               model_type=2, 
                               detector=AIModel.ModelDetectorChoices.NONE,
                               corrector=AIModel.ModelCorrectorChoices.NONE)

doc = Document.objects.create(user=User.objects.get(pk=1),
                              name="Test doc")

for i in range(1, 8):
    page = Page.objects.create(document=doc,
                           page_num=i,
                           is_demo=True,
                           image=f"images/test_images/{i}.jpg",)

