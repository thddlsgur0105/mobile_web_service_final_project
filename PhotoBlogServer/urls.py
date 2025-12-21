"""
URL configuration for PhotoBlogServer project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import routers
from blog.views import PostViewSet, WellbeingLogViewSet, register_user, logout_api, wellbeing_diagnosis, wellbeing_chat, emotion_chart_data, emotion_change_analysis, wellbeing_radar_chart, cluster_analysis, key_images_selection
from rest_framework.authtoken.views import obtain_auth_token

router = routers.DefaultRouter()
router.register(r'Post', PostViewSet, basename='post')
router.register(r'WellbeingLog', WellbeingLogViewSet, basename='wellbeinglog')

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("blog.urls")),  # blog 앱의 URL들을 포함
    path('api_root/', include(router.urls)),
    path('api_root/register/', register_user, name='api_register'),  # 회원가입 API
    path('api_root/logout/', logout_api, name='api_logout'),  # 로그아웃 API
    path('api_root/wellbeing-diagnosis/', wellbeing_diagnosis, name='wellbeing_diagnosis'),  # 웰빙 진단 API
    path('api_root/wellbeing-chat/', wellbeing_chat, name='wellbeing_chat'),  # 웰빙 대화 API
    path('api_root/emotion-chart-data/', emotion_chart_data, name='emotion_chart_data'),  # 감정 차트 데이터 API
    path('api_root/emotion-change-analysis/', emotion_change_analysis, name='emotion_change_analysis'),  # 감정 변화 분석 API
    path('api_root/wellbeing-radar-chart/', wellbeing_radar_chart, name='wellbeing_radar_chart'),  # 웰빙 Radar Chart API
    path('api_root/cluster-analysis/', cluster_analysis, name='cluster_analysis'),  # 클러스터 분석 API
    path('api_root/key-images-selection/', key_images_selection, name='key_images_selection'),  # 핵심 이미지 선정 API
    path('api_root/api-token-auth/', obtain_auth_token, name='api_login'),  # 로그인 API
]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)