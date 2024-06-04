import string
import random
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.contrib.auth import login, logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.template.loader import render_to_string

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny

from drf_spectacular.utils import extend_schema

from .serializers import *



def get_csrf_token(request):
    return JsonResponse({'csrftoken': get_token(request)})

class RegisterView(APIView):
    serializer_class = RegisterSerializer

    @extend_schema(responses=UserResponseSerializer)
    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            response_serializer = UserResponseSerializer(data={"detail": "User registered successfully"})
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.validated_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    serializer_class = LoginSerializer

    @extend_schema(responses=LoginResponseSerializer)
    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data
            if user is not None:
                login(request, user)
                response_serializer = LoginResponseSerializer(data={"detail": "Successfully logged in", "redirect_url": "/admin" if user.is_staff else "/profile"})
                response_serializer.is_valid(raise_exception=True)
                return Response(response_serializer.validated_data, status=status.HTTP_200_OK)
            response_serializer = LoginResponseSerializer(data={"detail": "Invalid credentials"})
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.validated_data, status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = UserSerializer

    def get(self, request, *args, **kwargs):
        user = request.user
        serializer = UserSerializer(user)
        return Response(serializer.data)

class ChangePasswordView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChangePasswordSerializer

    @extend_schema(responses=UserResponseSerializer)
    def post(self, request, *args, **kwargs):
        serializer = ChangePasswordSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user
            if not user.check_password(serializer.data['old_password']):
                response_serializer = UserResponseSerializer(data={"detail": "Wrong password"})
                response_serializer.is_valid(raise_exception=True)
                return Response(response_serializer.validated_data, status=status.HTTP_400_BAD_REQUEST)
            user.set_password(serializer.data['new_password'])
            user.save()
            update_session_auth_hash(request, user)  # Обновление сессии для предотвращения выхода пользователя
            response_serializer = UserResponseSerializer(data={"detail": "Password updated successfully"})
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.validated_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UpdateUserView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = UpdateUserSerializer

    @extend_schema(responses=UserResponseSerializer)
    def put(self, request, *args, **kwargs):
        user = request.user
        serializer = UpdateUserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            update_session_auth_hash(request, serializer.validated_data)
            response_serializer = UserResponseSerializer(data={"detail": "User information updated successfully"})
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.validated_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LogoutView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        logout(request)
        return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)

class PasswordResetRequestView(APIView):
    permission_classes = [AllowAny]
    serializer_class = ResetPasswordSerializer

    def generate_random_password(self, length=8):
        if length < 8:
            length = 8

        # Ensure the password has at least one digit and one special character
        digits = string.digits
        special_characters = string.punctuation
        letters = string.ascii_letters

        # Randomly choose one digit and one special character
        random_digit = random.choice(digits)
        random_special = random.choice(special_characters)

        # Generate the remaining characters
        remaining_length = length - 2
        random_chars = ''.join(random.choice(letters + digits + special_characters) for i in range(remaining_length))

        # Combine all parts and shuffle them to ensure randomness
        password = list(random_digit + random_special + random_chars)
        random.shuffle(password)

        return ''.join(password)

    @extend_schema(responses=UserResponseSerializer)
    def post(self, request):         
        serializer = ResetPasswordSerializer(data=request.data)
        
        if serializer.is_valid():
            user = get_user_model().objects.filter(email=serializer.validated_data['email']).first()
            if user:
                new_password = self.generate_random_password()
                user.set_password(new_password)
                user.save()

                context = {
                    'username': user.get_full_name(),
                    'new_password': new_password,
                    'protocol': settings.EMAIL_PROTOCOL,
                    'domain': settings.EMAIL_DOMAIN,
                }
                subject = f'Восстановление пароля | {settings.SITE_NAME}'
                email_template_name = 'password_reset.html'
                email_body = render_to_string(email_template_name, context)

                send_mail(subject, email_body, settings.DEFAULT_FROM_EMAIL, [serializer.validated_data['email']], fail_silently=False)

                response_serializer = UserResponseSerializer(data={"detail": "Password reset email has been sent"})
                response_serializer.is_valid(raise_exception=True)
                return Response(response_serializer.validated_data, status=status.HTTP_200_OK)
            else:
                response_serializer = UserResponseSerializer(data={"detail": "Invalid email"})
                response_serializer.is_valid(raise_exception=True)
                return Response(response_serializer.validated_data, status=status.HTTP_400_BAD_REQUEST)
        
        response_serializer = UserResponseSerializer(data={"detail": "Email is required"})
        response_serializer.is_valid(raise_exception=True)
        return Response(response_serializer.validated_data, status=status.HTTP_400_BAD_REQUEST)
