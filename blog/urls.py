from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register('Post', views.blogImage)

urlpatterns = [
    path("", views.post_list, name="post_list"),
    path("login/", auth_views.LoginView.as_view(template_name='blog/login.html'), name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("logout/success/", views.logout_success_view, name="logout_success"),
    path("register/", views.register_view, name="register"),
]