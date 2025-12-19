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
from blog.views import PostViewSet, WellbeingLogViewSet, register_user, logout_api
from rest_framework.authtoken.views import obtain_auth_token

router = routers.DefaultRouter()
router.register(r'Post', PostViewSet)
router.register(r'WellbeingLog', WellbeingLogViewSet)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("blog.urls")),  # blog 앱의 URL들을 포함
    path('api_root/', include(router.urls)),
    path('api_root/register/', register_user, name='api_register'),  # 회원가입 API
    path('api_root/logout/', logout_api, name='api_logout'),  # 로그아웃 API
    path('api-token-auth/', obtain_auth_token, name='api_login'),  # 로그인 API
]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)