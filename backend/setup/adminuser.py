# -*- coding: utf-8 -*-
from core.models import User
from django.contrib.auth.models import Group

userGroup, created_userGroup = Group.objects.get_or_create(name='user')
librarianGroup, created_librarianGroup = Group.objects.get_or_create(name='librarian')
moderatorGroup, created_moderatorGroup = Group.objects.get_or_create(name='moderator')
adminGroup, created_adminGroup = Group.objects.get_or_create(name='admin')


superuser = User.objects.create_superuser('superuser@gmail.com', 
                              'superuser', 
                              first_name='superuser',
                              last_name='superuser',
                              middle_name='superuser')
admin = User.objects.create_user('admin@gmail.com', 
                              'admin', 
                              first_name='admin',
                              last_name='admin',
                              middle_name='admin',
                              is_staff=True)
moderator = User.objects.create_user('moderator@gmail.com', 
                              'moderator', 
                              first_name='moderator',
                              last_name='moderator',
                              middle_name='moderator',
                              is_staff=True)
librarian = User.objects.create_user('librarian@gmail.com', 
                              'librarian', 
                              first_name='librarian',
                              last_name='librarian',
                              middle_name='librarian',
                              is_staff=True)
user = User.objects.create_user('user@gmail.com', 
                              'user', 
                              first_name='user',
                              last_name='user',
                              middle_name='user')

admin.groups.add(adminGroup)
moderator.groups.add(moderatorGroup)
librarian.groups.add(librarianGroup)
user.groups.add(userGroup)