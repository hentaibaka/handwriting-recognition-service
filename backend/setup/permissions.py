# -*- coding: utf-8 -*-
from django.contrib.auth.models import Group, Permission


userGroup, created_userGroup = Group.objects.get_or_create(name='user')
librarianGroup, created_librarianGroup = Group.objects.get_or_create(name='librarian')
moderatorGroup, created_moderatorGroup = Group.objects.get_or_create(name='moderator')
adminGroup, created_adminGroup = Group.objects.get_or_create(name='admin')

add_document = Permission.objects.get(codename='add_document')
change_document = Permission.objects.get(codename='change_document')
delete_document = Permission.objects.get(codename='delete_document')
view_document = Permission.objects.get(codename='view_document')

add_page = Permission.objects.get(codename='add_page')
change_page = Permission.objects.get(codename='change_page')
delete_page = Permission.objects.get(codename='delete_page')
view_page = Permission.objects.get(codename='view_page')

add_string = Permission.objects.get(codename='add_string')
change_string = Permission.objects.get(codename='change_string')
delete_string = Permission.objects.get(codename='delete_string')
view_string = Permission.objects.get(codename='view_string')

add_aimodel = Permission.objects.get(codename='add_aimodel')
change_aimodel = Permission.objects.get(codename='change_aimodel')
delete_aimodel = Permission.objects.get(codename='delete_aimodel')
view_aimodel = Permission.objects.get(codename='view_aimodel')

add_train = Permission.objects.get(codename='add_train')
change_train = Permission.objects.get(codename='change_train')
delete_train = Permission.objects.get(codename='delete_train')
view_train = Permission.objects.get(codename='view_train')

add_dataset = Permission.objects.get(codename='add_dataset')
change_dataset = Permission.objects.get(codename='change_dataset')
delete_dataset = Permission.objects.get(codename='delete_dataset')
view_dataset = Permission.objects.get(codename='view_dataset')

add_metric = Permission.objects.get(codename='add_metric')
change_metric = Permission.objects.get(codename='change_metric')
delete_metric = Permission.objects.get(codename='delete_metric')
view_metric = Permission.objects.get(codename='view_metric')

add_user = Permission.objects.get(codename='add_user')
change_user = Permission.objects.get(codename='change_user')
delete_user = Permission.objects.get(codename='delete_user')
view_user = Permission.objects.get(codename='view_user')

userPermList = [
    change_user,
    view_user,
]
librarianPermList = [
    add_document,
    change_document,
    delete_document,
    view_document,

    add_page,
    change_page,
    delete_page,
    view_page,

    add_string,
    change_string,
    delete_string,
    view_string,
    
    add_dataset,
    change_dataset,
    view_dataset,

    add_train,
    change_train,
    view_train,

    view_metric,
]
moderatorPermList = [
    view_aimodel,
    change_aimodel,

    change_metric,


]
adminPermList = [
    add_user,
    delete_user,
]

userGroup.permissions.set(userPermList)
librarianGroup.permissions.set(librarianPermList + userPermList)
moderatorGroup.permissions.set(moderatorPermList + librarianPermList + userPermList)
adminGroup.permissions.set(adminPermList + moderatorPermList + librarianPermList + userPermList)
