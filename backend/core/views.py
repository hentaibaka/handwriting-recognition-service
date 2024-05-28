from documents import serializers
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import login, authenticate
from .serializers import LoginSerializer
from drf_spectacular.utils import extend_schema


class LoginView(APIView):
    serializer_class = LoginSerializer

    @extend_schema(responses=str)
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data
        login(request, user)
        return Response({"detail": "Successfully logged in.", "redirect_url": "/admin"}, status=status.HTTP_200_OK)
