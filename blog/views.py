from django.shortcuts import render
from .models import Post
from django.utils import timezone
from django.db.models import Q
from django.contrib.auth import get_user_model

from rest_framework import viewsets
from .serializers import PostSerializer

def post_list(request):
    posts = Post.objects.filter(
        Q(published_date__lte=timezone.now()) | Q(published_date__isnull=True)
    ).order_by('-published_date')
    return render(request, "blog/post_list.html", {'posts': posts})

class blogImage(viewsets.ModelViewSet):
 queryset = Post.objects.all()
 serializer_class = PostSerializer

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all().order_by('-published_date')
    serializer_class = PostSerializer

    def perform_create(self, serializer):
        author = serializer.validated_data.get('author')
        if author is None:
            if self.request.user and self.request.user.is_authenticated:
                author = self.request.user
            else:
                User = get_user_model()
                author = User.objects.first()

        published_date = serializer.validated_data.get('published_date')
        if published_date is None:
            published_date = timezone.now()

        serializer.save(author=author, published_date=published_date)