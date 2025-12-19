from blog.models import Post, WellbeingLog
from rest_framework import serializers
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

class PostSerializer(serializers.HyperlinkedModelSerializer):
    author = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(), required=False)

    class Meta:
        model = Post
        fields = '__all__'


class WellbeingLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = WellbeingLog
        fields = '__all__'


class UserRegistrationSerializer(serializers.ModelSerializer):
    """회원가입을 위한 시리얼라이저"""
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True, min_length=8)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm')
    
    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("비밀번호가 일치하지 않습니다.")
        return data
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password']
        )
        # 토큰 자동 생성
        Token.objects.create(user=user)
        return user