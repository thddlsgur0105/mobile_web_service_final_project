# Create your models here.

from django.conf import settings
from django.db import models
from django.utils import timezone


class WellbeingLog(models.Model):
    """
    YOLOv5 + 표정/움직임 분석 결과를 주기적으로 저장하는 모델.
    하루/주간 웰빙 리포트를 만들 때 이 데이터를 집계해서 사용합니다.
    """

    created_at = models.DateTimeField(default=timezone.now)
    dominant_emotion = models.CharField(max_length=50)
    dominant_emotion_ratio = models.FloatField()
    emotion_counts = models.JSONField()  # {"happy": 10, "sad": 5, ...}
    avg_movement = models.FloatField()

    def __str__(self):
        return f"{self.created_at:%Y-%m-%d %H:%M} - {self.dominant_emotion}"


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