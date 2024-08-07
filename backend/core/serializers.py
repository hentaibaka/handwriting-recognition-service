from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from .models import *
from django.contrib.auth.models import Group


class ResetPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()

class UserResponseSerializer(serializers.Serializer):
    detail = serializers.CharField()

class LoginResponseSerializer(UserResponseSerializer):
    redirect_url = serializers.CharField(allow_blank=True)

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = get_user_model()
        fields = ('first_name', 'last_name', 'middle_name', 'email', 'password')

    def create(self, validated_data):
        userGroup = Group.objects.get(name='user')
        User = get_user_model()
        user = User.objects.create_user(
            email=validated_data['email'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
            middle_name=validated_data['middle_name'],
            password=validated_data['password'],
            is_staff=False,
        )
        user.groups.add(userGroup)
        return user

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Invalid credentials")

class UserSerializer(serializers.ModelSerializer):
    status = serializers.SerializerMethodField('get_status')

    def get_status(self, obj: User) -> str:
        groups_query = obj.groups.all()
        if groups_query.exists():   
            groups = groups_query.values_list('name', flat=True)
            if 'user' in groups:
                status = 'Пользователь'
            if 'librarian' in groups:
                status = 'Библиатекарь'
            if 'moderator' in groups:
                status = 'Модератор'
            if 'admin' in groups:
                status = 'Администратор'
            if obj.is_superuser:
                status = 'Суперпользователь'
        else:
            status = '-'
        return status
    class Meta:
        model = get_user_model()
        fields = ['id', 'first_name', 'last_name', 'middle_name', 'email', 'status',]

class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True, validators=[validate_password])

class UpdateUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ['first_name', 'last_name', 'middle_name', 'email']
