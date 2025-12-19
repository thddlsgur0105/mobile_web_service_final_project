from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import Post, WellbeingLog
from django.utils import timezone
from django.db.models import Q, Avg
from django.contrib.auth import get_user_model
from datetime import timedelta

from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authtoken.models import Token
from .serializers import PostSerializer, WellbeingLogSerializer, UserRegistrationSerializer

@login_required
def post_list(request):
    """
    포스트 목록을 보여주는 뷰.
    로그인이 필수입니다.
    """
    posts = Post.objects.filter(
        Q(published_date__lte=timezone.now()) | Q(published_date__isnull=True)
    ).order_by('-published_date')
    return render(request, "blog/post_list.html", {'posts': posts})


def register_view(request):
    """
    회원가입 뷰.
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'계정이 생성되었습니다: {username}')
            # 자동 로그인
            login(request, user)
            return redirect('post_list')
    else:
        form = UserCreationForm()
    return render(request, 'blog/register.html', {'form': form})


def logout_view(request):
    """
    로그아웃 뷰.
    Django 세션 로그아웃 및 localhost의 auth_info.json 삭제 안내.
    """
    from django.contrib.auth import logout
    
    # Django 세션 로그아웃
    logout(request)
    messages.success(request, '로그아웃되었습니다.')
    
    # 로그아웃 완료 페이지로 리다이렉트
    return redirect('logout_success')


def logout_success_view(request):
    """
    로그아웃 완료 페이지.
    localhost의 auth_info.json 삭제 안내.
    """
    return render(request, 'blog/logout_success.html')

class blogImage(viewsets.ModelViewSet):
 queryset = Post.objects.all()
 serializer_class = PostSerializer

class PostViewSet(viewsets.ModelViewSet):
    """
    게시를 위한 ViewSet.
    인증된 사용자만 접근 가능합니다.
    """
    queryset = Post.objects.all().order_by('-published_date')
    serializer_class = PostSerializer
    permission_classes = [IsAuthenticated]  # 인증 필수

    def perform_create(self, serializer):
        # 인증된 사용자를 작성자로 설정
        author = serializer.validated_data.get('author')
        if author is None:
            author = self.request.user

        published_date = serializer.validated_data.get('published_date')
        if published_date is None:
            published_date = timezone.now()

        serializer.save(author=author, published_date=published_date)


class WellbeingLogViewSet(viewsets.ModelViewSet):
    """
    YOLOv5/DeepFace 분석 결과를 저장·조회하기 위한 ViewSet.
    클라이언트(분석 스크립트)가 주기적으로 POST로 데이터를 전송합니다.
    인증된 사용자만 접근 가능합니다.
    """

    queryset = WellbeingLog.objects.all().order_by('-created_at')
    serializer_class = WellbeingLogSerializer
    permission_classes = [IsAuthenticated]  # 인증 필수

    @action(detail=False, methods=['get'])
    def summary(self, request):
        """
        최근 웰빙 로그를 집계해 요약 정보를 반환합니다.
        - today: 오늘 0시 이후 데이터 집계
        - last_7_days: 최근 7일 데이터 집계
        """
        now = timezone.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        last_7_days_start = now - timedelta(days=7)

        qs_today = self.queryset.filter(created_at__gte=today_start)
        qs_week = self.queryset.filter(created_at__gte=last_7_days_start)

        def aggregate_summary(qs):
            count = qs.count()
            if count == 0:
                return {
                    "count": 0,
                    "dominant_emotion": None,
                    "emotion_counts": {},
                    "avg_movement": 0.0,
                }

            # emotion_counts 필드를 모두 합산
            emotion_totals = {}
            for log in qs:
                counts = log.emotion_counts or {}
                for k, v in counts.items():
                    emotion_totals[k] = emotion_totals.get(k, 0) + int(v)

            dominant_emotion = None
            if emotion_totals:
                dominant_emotion = max(emotion_totals.items(), key=lambda x: x[1])[0]

            avg_movement = qs.aggregate(avg_movement=Avg("avg_movement"))[
                "avg_movement"
            ] or 0.0

            return {
                "count": count,
                "dominant_emotion": dominant_emotion,
                "emotion_counts": emotion_totals,
                "avg_movement": avg_movement,
            }

        data = {
            "today": aggregate_summary(qs_today),
            "last_7_days": aggregate_summary(qs_week),
        }

        return Response(data)


@api_view(['POST'])
@permission_classes([AllowAny])
def logout_api(request):
    """
    Edge System용 로그아웃 API.
    토큰을 삭제하여 로그아웃 처리.
    POST /api_root/logout/
    Headers: Authorization: Token <token>
    """
    if request.user.is_authenticated:
        # 토큰 삭제
        try:
            token = Token.objects.get(user=request.user)
            token.delete()
        except Token.DoesNotExist:
            pass
        
        return Response({
            'message': '로그아웃되었습니다. auth_info.json 파일을 삭제하세요.',
            'auth_info_file': '프로젝트 루트의 auth_info.json 파일을 삭제하세요.'
        }, status=status.HTTP_200_OK)
    else:
        return Response({
            'message': '이미 로그아웃된 상태입니다.'
        }, status=status.HTTP_200_OK)


@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def register_user(request):
    """
    회원가입 API
    GET /api_root/register/ - 회원가입 폼 정보 반환 (웹 브라우저 접근용)
    POST /api_root/register/ - 회원가입 처리
    {
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123",
        "password_confirm": "password123"
    }
    """
    if request.method == 'GET':
        # GET 요청 시 회원가입 폼 정보 반환 (웹 브라우저 접근용)
        return Response({
            'message': '회원가입 API',
            'method': 'POST',
            'fields': {
                'username': '사용자명 (필수)',
                'email': '이메일 (선택사항)',
                'password': '비밀번호 (필수, 최소 8자)',
                'password_confirm': '비밀번호 확인 (필수)'
            },
            'web_form_url': '/register/',
            'note': '웹 브라우저에서 회원가입하려면 /register/ 페이지를 사용하세요.'
        }, status=status.HTTP_200_OK)
    
    # POST 요청 처리
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'message': '회원가입이 완료되었습니다.',
            'username': user.username,
            'token': token.key
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)