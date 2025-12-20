# Create your models here.

from django.conf import settings
from django.db import models
from django.utils import timezone


class WellbeingLog(models.Model):
    """
    YOLOv5 + 표정/움직임 분석 결과를 주기적으로 저장하는 모델.
    하루/주간 웰빙 리포트를 만들 때 이 데이터를 집계해서 사용합니다.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='wellbeing_logs', null=True, blank=True
    )
    created_at = models.DateTimeField(default=timezone.now)
    dominant_emotion = models.CharField(max_length=50)
    dominant_emotion_ratio = models.FloatField()
    emotion_counts = models.JSONField()  # {"happy": 10, "sad": 5, ...}
    avg_movement = models.FloatField()
    
    # 새로운 필드들
    pose = models.CharField(max_length=20, blank=True, null=True)  # sitting, standing, bending
    head_pose = models.JSONField(blank=True, null=True)  # {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    eye_blink_count = models.IntegerField(default=0)  # 눈 깜빡임 횟수
    focus_level = models.FloatField(default=0.0)  # 집중도 (0.0 ~ 1.0)
    fatigue_level = models.FloatField(default=0.0)  # 피로도 (0.0 ~ 1.0)
    
    # 사용자 이미지 필드 추가
    image = models.ImageField(upload_to='wellbeing_images/%Y/%m/%d/', blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} - {self.created_at:%Y-%m-%d %H:%M} - {self.dominant_emotion}"


class Post(models.Model):
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE
    )
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(default=timezone.now)
    published_date = models.DateTimeField(blank=True, null=True)
    image = models.ImageField(upload_to='blog_image/%Y/%m/%d/', default="blog_image/default_error.png", blank=True, null=True)
    
    def publish(self):
        self.published_date = timezone.now()
        self.save()
    
    def __str__(self):
        return self.title