# -*- coding: utf-8 -*-
from ai_service.models import *
from documents.models import *
from django.contrib.auth import get_user_model

User = get_user_model()

model = AIModel.objects.create(name="best_accuracy",
                               is_current=False)
model = AIModel.objects.create(name="best_accuracy2",
                               is_current=True)

doc = Document.objects.create(user=User.objects.get(pk=1),
                              name="Test doc")

page = Page.objects.create(document=doc,
                           page_num=1,
                           is_demo=True,
                           image="images/test_images/test1.jpg",)
page = Page.objects.create(document=doc,
                           page_num=2,
                           is_demo=True,
                           image="images/test_images/test2.jpg",)
page = Page.objects.create(document=doc,
                           page_num=3,
                           is_demo=True,
                           image="images/test_images/test3.jpg",)
page = Page.objects.create(document=doc,
                           page_num=4,
                           is_demo=True,
                           image="images/test_images/test4(bad_image).jpg",)
page = Page.objects.create(document=doc,
                           page_num=5,
                           is_demo=True,
                           image="images/test_images/test5.jpg",)
