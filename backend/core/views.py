from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.contrib.auth import login, logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.core.mail import send_mail
from django.conf import settings

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

from drf_spectacular.utils import extend_schema

from .serializers import *



def get_csrf_token(request):
    return JsonResponse({'csrftoken': get_token(request)})

#@method_decorator(csrf_exempt, name='dispatch')
class RegisterView(APIView):
    serializer_class = RegisterSerializer
    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({'detail': 'User registered successfully'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#@method_decorator(csrf_exempt, name='dispatch')
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

    @extend_schema(responses=UserResponseSerializer)
    def post(self, request):
        logout(request)
        response_serializer = UserResponseSerializer(data={"detail": "Successfully logged out"})
        response_serializer.is_valid(raise_exception=True)
        return Response(response_serializer.validated_data, status=status.HTTP_200_OK)

class PasswordResetView(APIView):
    permission_classes = []
    serializer_class = PasswordResetSerializer

    @extend_schema(responses=UserResponseSerializer)
    def post(self, request):
        serializer = PasswordResetSerializer(data=request.data)
        if not serializer.is_valid():
            response_serializer = UserResponseSerializer(data={"detail": "Email is required"})
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.validated_data, status=status.HTTP_400_BAD_REQUEST)
        try:
            User = get_user_model()
            user = User.objects.get(email=serializer.data['email'])
        except User.DoesNotExist:
            response_serializer = UserResponseSerializer(data={"detail": "User with this email does not exist"})
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.validated_data, status=status.HTTP_404_NOT_FOUND)

        # Create password reset link
        reset_link = request.build_absolute_uri(reverse('password_reset_confirm'))
        send_mail(
            'Password Reset',
            f'Use the following link to reset your password: {reset_link}',
            settings.DEFAULT_FROM_EMAIL,
            [serializer.data['email']],
            fail_silently=False,
        )
        response_serializer = UserResponseSerializer(data={"detail": "Password reset email has been sent"})
        response_serializer.is_valid(raise_exception=True)
        return Response(response_serializer.validated_data, status=status.HTTP_200_OK)

