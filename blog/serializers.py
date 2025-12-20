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
    
    def to_internal_value(self, data):
        """
        multipart/form-data에서 JSON 필드를 문자열로 받을 때 자동으로 파싱합니다.
        """
        import json
        from django.http import QueryDict
        
        # data를 mutable한 dict로 변환
        if isinstance(data, QueryDict):
            data_dict = data.dict()
        else:
            data_dict = dict(data) if hasattr(data, '__iter__') and not isinstance(data, str) else data
        
        # emotion_counts가 문자열로 오면 JSON으로 파싱
        if 'emotion_counts' in data_dict:
            emotion_counts_value = data_dict.get('emotion_counts')
            if isinstance(emotion_counts_value, str):
                try:
                    if emotion_counts_value.strip():  # 빈 문자열이 아닐 때만 파싱
                        data_dict['emotion_counts'] = json.loads(emotion_counts_value)
                    else:
                        data_dict['emotion_counts'] = {}
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    # 파싱 실패 시 빈 dict로 설정
                    data_dict['emotion_counts'] = {}
        
        # head_pose가 문자열로 오면 JSON으로 파싱
        if 'head_pose' in data_dict:
            head_pose_value = data_dict.get('head_pose')
            if isinstance(head_pose_value, str):
                try:
                    if head_pose_value.strip():  # 빈 문자열이 아닐 때만 파싱
                        data_dict['head_pose'] = json.loads(head_pose_value)
                    else:
                        data_dict['head_pose'] = None
                except (json.JSONDecodeError, TypeError, ValueError):
                    data_dict['head_pose'] = None
        
        return super().to_internal_value(data_dict)


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