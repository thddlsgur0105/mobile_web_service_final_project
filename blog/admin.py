from django.contrib import admin
from .models import Post

# Register your models here.

admin.site.register(Post) # 관리자 페이지에 Post 모델을 등록