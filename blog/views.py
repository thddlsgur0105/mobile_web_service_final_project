from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import Post, WellbeingLog
from django.utils import timezone
from django.db.models import Q, Avg
from django.contrib.auth import get_user_model
from django.conf import settings
from datetime import timedelta
import os
import base64
import requests
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

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
    로그인이 필수이며, 현재 로그인한 사용자의 포스트만 표시합니다.
    """
    posts = Post.objects.filter(
        author=request.user
    ).filter(
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
    인증된 사용자만 접근 가능하며, 자신의 포스트만 조회할 수 있습니다.
    """
    serializer_class = PostSerializer
    permission_classes = [IsAuthenticated]  # 인증 필수

    def get_queryset(self):
        """
        현재 로그인한 사용자의 포스트만 반환합니다.
        """
        return Post.objects.filter(author=self.request.user).order_by('-published_date')

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
    인증된 사용자만 접근 가능하며, 자신의 로그만 조회할 수 있습니다.
    """

    serializer_class = WellbeingLogSerializer
    permission_classes = [IsAuthenticated]  # 인증 필수

    def get_queryset(self):
        """
        현재 로그인한 사용자의 웰빙 로그만 반환합니다.
        user가 null인 기존 데이터는 제외합니다.
        """
        return WellbeingLog.objects.filter(user=self.request.user).exclude(user__isnull=True).order_by('-created_at')

    def perform_create(self, serializer):
        """
        웰빙 로그 생성 시 현재 로그인한 사용자를 자동으로 설정합니다.
        """
        serializer.save(user=self.request.user)

    @action(detail=False, methods=['get'])
    def summary(self, request):
        """
        최근 웰빙 로그를 집계해 요약 정보를 반환합니다.
        - today: 오늘 0시 이후 데이터 집계
        - last_7_days: 최근 7일 데이터 집계
        현재 로그인한 사용자의 데이터만 집계합니다.
        """
        now = timezone.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        last_7_days_start = now - timedelta(days=7)

        # 현재 사용자의 로그만 필터링
        user_queryset = self.get_queryset()
        qs_today = user_queryset.filter(created_at__gte=today_start)
        qs_week = user_queryset.filter(created_at__gte=last_7_days_start)

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


def analyze_user_wellbeing(user):
    """
    사용자의 웰빙 데이터를 분석하여 진단 리포트를 생성합니다.
    """
    # 최근 웰빙 로그 가져오기
    wellbeing_logs = WellbeingLog.objects.filter(user=user).order_by('-created_at')[:30]
    
    # 최근 포스트 가져오기 (이미지 포함)
    recent_posts = Post.objects.filter(author=user).order_by('-created_date')[:20]
    
    # 감정 통계
    emotion_counts = {}
    total_movement = 0.0
    movement_count = 0
    
    for log in wellbeing_logs:
        if log.emotion_counts:
            for emotion, count in log.emotion_counts.items():
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + count
        if log.avg_movement:
            total_movement += log.avg_movement
            movement_count += 1
    
    avg_movement = total_movement / movement_count if movement_count > 0 else 0.0
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
    
    # 분석 데이터 구성
    analysis_data = {
        "total_logs": wellbeing_logs.count(),
        "total_posts": recent_posts.count(),
        "emotion_distribution": emotion_counts,
        "dominant_emotion": dominant_emotion,
        "avg_movement": round(avg_movement, 2),
        "recent_activity": wellbeing_logs.count() > 0,
    }
    
    return analysis_data


def get_llm_response(prompt, system_prompt=None, use_openai=True):
    """
    LLM API를 호출하여 응답을 받습니다.
    OpenAI API 또는 다른 LLM 서비스를 사용할 수 있습니다.
    """
    # OpenAI API 키 확인
    api_key = getattr(settings, 'OPENAI_API_KEY', None) or os.environ.get('OPENAI_API_KEY')
    print(api_key)

    if not api_key and use_openai:
        # API 키가 없으면 기본 응답 반환
        return "LLM 서비스가 설정되지 않았습니다. OPENAI_API_KEY 환경 변수를 설정해주세요."
    
    try:
        if use_openai and api_key:
            # OpenAI API 사용
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": "gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"LLM API 오류: {response.status_code}"
        else:
            # 기본 응답 (LLM 없이)
            return "LLM 서비스가 활성화되지 않았습니다."
            
    except Exception as e:
        return f"LLM 호출 중 오류 발생: {str(e)}"


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def emotion_chart_data(request):
    """
    사용자 감정 변화 차트 데이터를 반환하는 API.
    - 과거 일주일: 7일 전 ~ 1시간 전
    - 최근 한 시간: 1시간 전 ~ 현재
    """
    user = request.user
    now = timezone.now()
    one_week_ago = now - timedelta(days=7)
    one_hour_ago = now - timedelta(hours=1)
    
    # 과거 일주일간 데이터 (7일 전 ~ 1시간 전, 최근 1시간 제외)
    logs_week = WellbeingLog.objects.filter(
        user=user,
        created_at__gte=one_week_ago,
        created_at__lt=one_hour_ago  # 1시간 전 이전 데이터만
    )
    
    # 최근 한 시간 데이터 (1시간 전 ~ 현재)
    logs_hour = WellbeingLog.objects.filter(
        user=user,
        created_at__gte=one_hour_ago,
        created_at__lte=now  # 현재까지 포함
    )
    
    def aggregate_emotions(logs):
        """로그에서 감정 빈도 집계"""
        emotion_counts = {}
        for log in logs:
            if log.emotion_counts:
                for emotion, count in log.emotion_counts.items():
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + int(count)
        return emotion_counts
    
    week_emotions = aggregate_emotions(logs_week)
    hour_emotions = aggregate_emotions(logs_hour)
    
    return Response({
        'week': week_emotions,
        'hour': hour_emotions,
        'week_total': logs_week.count(),
        'hour_total': logs_hour.count(),
        'week_range': {
            'start': one_week_ago.isoformat(),
            'end': one_hour_ago.isoformat()
        },
        'hour_range': {
            'start': one_hour_ago.isoformat(),
            'end': now.isoformat()
        }
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def emotion_change_analysis(request):
    """
    과거 일주일 대비 최근 한 시간 동안의 감정 변화를 분석하는 API.
    LLM을 사용하여 짧은 한 줄 요약과 추천을 제공합니다.
    """
    user = request.user
    now = timezone.now()
    one_week_ago = now - timedelta(days=7)
    one_hour_ago = now - timedelta(hours=1)
    
    # 과거 일주일간 데이터 (7일 전 ~ 1시간 전)
    logs_week = WellbeingLog.objects.filter(
        user=user,
        created_at__gte=one_week_ago,
        created_at__lt=one_hour_ago
    )
    
    # 최근 한 시간 데이터 (1시간 전 ~ 현재)
    logs_hour = WellbeingLog.objects.filter(
        user=user,
        created_at__gte=one_hour_ago,
        created_at__lte=now
    )
    
    def aggregate_emotions(logs):
        """로그에서 감정 빈도 집계"""
        emotion_counts = {}
        for log in logs:
            if log.emotion_counts:
                for emotion, count in log.emotion_counts.items():
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + int(count)
        return emotion_counts
    
    week_emotions = aggregate_emotions(logs_week)
    hour_emotions = aggregate_emotions(logs_hour)
    
    # 데이터가 없으면 기본 메시지 반환
    if not week_emotions and not hour_emotions:
        return Response({
            'analysis': '아직 충분한 데이터가 없습니다. 조금 더 기다려주세요.',
            'week_emotions': {},
            'hour_emotions': {}
        }, status=status.HTTP_200_OK)
    
    # LLM 프롬프트 생성
    system_prompt = """당신은 감정 분석 전문가입니다. 사용자의 감정 변화를 분석하고 간단하고 실용적인 조언을 제공합니다.

답변 규칙:
1. 반드시 두 줄로만 답변 (각 줄 최대 50자)
2. 첫 번째 줄: 감정 변화 요약
3. 두 번째 줄: 추천 문구
4. "→" 기호를 사용하지 않음
5. 한국어로 자연스럽게 작성
6. 긍정적이고 격려하는 톤 사용

예시:
"과거 대비 행복한 감정이 30% 증가했어요
이 좋은 기분을 유지하기 위해 산책을 추천드려요"

"최근 스트레스가 증가한 것 같아요
잠시 휴식을 취하며 심호흡을 해보세요"
"""
    
    user_prompt = f"""과거 일주일간 감정 분포:
{json.dumps(week_emotions, ensure_ascii=False)}

최근 한 시간 감정 분포:
{json.dumps(hour_emotions, ensure_ascii=False)}

위 데이터를 비교하여 감정 변화를 분석하고, 한 줄로 요약과 추천을 제공해주세요."""
    
    # LLM 호출
    analysis = get_llm_response(user_prompt, system_prompt)
    
    return Response({
        'analysis': analysis,
        'week_emotions': week_emotions,
        'hour_emotions': hour_emotions
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def wellbeing_radar_chart(request):
    """
    웰빙 데이터를 클러스터링하여 Radar Chart 데이터를 반환하는 API.
    Movement, Pose, Focus, Fatigue 값을 정규화하고 클러스터링하여 종합 웰빙 상태를 시각화합니다.
    """
    user = request.user
    
    # 모든 WellbeingLog 데이터 가져오기 (과거부터 현재까지)
    logs = WellbeingLog.objects.filter(user=user).order_by('created_at')
    
    if logs.count() < 3:
        return Response({
            'error': '데이터가 부족합니다. 최소 3개 이상의 데이터가 필요합니다.',
            'data': None
        }, status=status.HTTP_200_OK)
    
    try:
        # 1. 기본 지표 추출 및 정규화
        features = []
        timestamps = []
        
        for log in logs:
            # Movement 정규화 (0~1)
            movement_rate = min(1.0, max(0.0, log.avg_movement / 100.0))  # 100을 최대값으로 가정
            
            # Pose 정규화 (sitting=0.0, standing=0.5, bending=1.0)
            pose_score = 0.5  # 기본값
            if log.pose == 'sitting':
                pose_score = 0.0
            elif log.pose == 'standing':
                pose_score = 0.5
            elif log.pose == 'bending':
                pose_score = 1.0
            
            # Head Pose 정규화 (각도 기반)
            posture_variance = 0.0
            if log.head_pose:
                pitch = abs(log.head_pose.get('pitch', 0))
                yaw = abs(log.head_pose.get('yaw', 0))
                roll = abs(log.head_pose.get('roll', 0))
                # 각도를 0~1로 정규화 (90도를 최대값으로 가정)
                posture_variance = min(1.0, (pitch + yaw + roll) / 270.0)
            
            # Focus 정규화 (이미 0~1)
            focus_level = min(1.0, max(0.0, log.focus_level))
            
            # Focus 안정성 (시간 분산은 나중에 계산)
            focus_stability = 1.0  # 기본값
            
            # Fatigue 정규화 (이미 0~1)
            fatigue_level = min(1.0, max(0.0, log.fatigue_level))
            
            # Feature 벡터 구성
            feature_vector = [
                movement_rate,
                pose_score,
                posture_variance,
                focus_level,
                focus_stability,
                fatigue_level
            ]
            
            features.append(feature_vector)
            timestamps.append(log.created_at.isoformat())
        
        # numpy 배열로 변환
        features_array = np.array(features)
        
        # 2. 시간 기반 분석 (Sliding Window)
        # Focus 안정성 계산 (시간 분산)
        window_size = min(5, len(features))  # 5분 윈도우 또는 전체 데이터
        for i in range(len(features)):
            start_idx = max(0, i - window_size + 1)
            window_focus = [f[3] for f in features[start_idx:i+1]]  # focus_level
            if len(window_focus) > 1:
                focus_stability = 1.0 - min(1.0, np.std(window_focus))  # 분산이 작을수록 안정적
            else:
                focus_stability = 1.0
            features[i][4] = focus_stability
        
        # Fatigue 기울기 계산 (Linear Regression)
        fatigue_trends = []
        for i in range(len(features)):
            start_idx = max(0, i - window_size + 1)
            window_fatigue = [f[5] for f in features[start_idx:i+1]]  # fatigue_level
            if len(window_fatigue) > 1:
                x = np.arange(len(window_fatigue))
                slope = np.polyfit(x, window_fatigue, 1)[0]  # 기울기
                fatigue_trend = min(1.0, max(0.0, (slope + 1) / 2))  # -1~1을 0~1로 정규화
            else:
                fatigue_trend = 0.5
            fatigue_trends.append(fatigue_trend)
        
        # 3. 클러스터링 (K-means)
        # 최적 클러스터 수 결정 (2~5개)
        n_clusters = min(5, max(2, len(features) // 3))
        
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 4. 클러스터 중심점 계산 (Radar Chart용)
        cluster_centers = kmeans.cluster_centers_
        
        # 역정규화하여 원래 스케일로 변환
        cluster_centers_original = scaler.inverse_transform(cluster_centers)
        
        # 5. 클러스터별 샘플 수 계산 및 정렬
        cluster_counts = []
        for i in range(n_clusters):
            count = int(np.sum(cluster_labels == i))
            cluster_counts.append({
                'index': i,
                'count': count,
                'data': cluster_centers_original[i].tolist()
            })
        
        # 샘플 수가 많은 순으로 정렬
        cluster_counts.sort(key=lambda x: x['count'], reverse=True)
        
        # 상위 3개 클러스터만 선택
        top_3_clusters = cluster_counts[:3]
        
        # 6. Radar Chart 데이터 구성
        labels = ['Movement', 'Pose', 'Posture Variance', 'Focus', 'Focus Stability', 'Fatigue']
        
        # 상위 3개 클러스터 데이터
        cluster_data = []
        for idx, cluster in enumerate(top_3_clusters):
            cluster_data.append({
                'label': f'Cluster {cluster["index"] + 1}',
                'data': cluster['data'],
                'count': cluster['count']
            })
        
        return Response({
            'labels': labels,
            'clusters': cluster_data,
            'total_samples': len(features),
            'n_clusters': len(top_3_clusters),
            'timestamps': timestamps
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        import traceback
        return Response({
            'error': f'클러스터링 오류: {str(e)}',
            'traceback': traceback.format_exc(),
            'data': None
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cluster_analysis(request):
    """
    클러스터 데이터를 분석하여 자연어로 설명하는 API.
    각 클러스터의 웰빙 상태를 한 줄로 요약합니다.
    """
    try:
        data = request.data
        cluster_data = data.get('cluster_data')
        cluster_label = data.get('cluster_label', 'Cluster')
        cluster_count = data.get('cluster_count', 0)
        
        if not cluster_data:
            return Response({
                'error': '클러스터 데이터가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 클러스터 데이터를 자연어로 변환
        labels = ['Movement', 'Pose', 'Posture Variance', 'Focus', 'Focus Stability', 'Fatigue']
        analysis_text = f"클러스터 분석:\n"
        for i, (label, value) in enumerate(zip(labels, cluster_data)):
            percentage = value * 100
            analysis_text += f"- {label}: {percentage:.1f}%\n"
        
        system_prompt = """당신은 웰빙 상태 분석 전문가입니다. 사용자의 웰빙 지표 데이터를 분석하여 한 줄로 간결하고 자연스러운 한국어 설명을 제공합니다.

답변 규칙:
1. 반드시 한 줄로만 답변 (최대 100자)
2. 웰빙 상태를 종합적으로 평가
3. Movement, Pose, Focus, Fatigue 등의 지표를 고려
4. 긍정적이고 격려하는 톤 사용
5. 구체적이고 실용적인 설명 제공"""
        
        user_prompt = f"""다음은 웰빙 상태 클러스터 데이터입니다 (총 {cluster_count}개 샘플):
{analysis_text}

이 데이터를 바탕으로 이 클러스터의 웰빙 상태를 한 줄로 자연스럽게 설명해주세요."""
        
        analysis = get_llm_response(user_prompt, system_prompt)
        
        return Response({
            'analysis': analysis,
            'cluster_label': cluster_label
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': f'분석 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def key_images_selection(request):
    """
    최근 7일 동안의 WellbeingLog 이미지 중에서 LLM이 Negative, Normal, Positive 상태의 이미지 3개를 선정합니다.
    """
    user = request.user
    now = timezone.now()
    seven_days_ago = now - timedelta(days=7)
    
    # 최근 7일 동안의 WellbeingLog 가져오기 (이미지가 있는 것만)
    logs = WellbeingLog.objects.filter(
        user=user,
        created_at__gte=seven_days_ago,
        image__isnull=False
    ).exclude(image='').order_by('-created_at')
    
    if logs.count() < 3:
        return Response({
            'error': f'최근 7일 동안의 이미지가 {logs.count()}개입니다. (필요: 3개 이상)',
            'images': [],
            'total_count': logs.count()
        }, status=status.HTTP_200_OK)
    
    # 각 로그의 지표 데이터와 이미지 URL 준비
    images_data = []
    for log in logs:
        # 이미지 URL 생성
        if log.image:
            image_url = request.build_absolute_uri(log.image.url)
        else:
            continue
        
        # 웰빙 점수 계산 (간단한 휴리스틱)
        # Positive: happy 감정, 높은 집중도, 낮은 피로도
        # Negative: sad/angry 감정, 낮은 집중도, 높은 피로도
        emotion_score = 0.0
        if log.dominant_emotion == 'happy':
            emotion_score = 1.0
        elif log.dominant_emotion in ['sad', 'angry', 'fear']:
            emotion_score = -1.0
        
        wellbeing_score = (
            emotion_score * 0.4 +
            log.focus_level * 0.3 -
            log.fatigue_level * 0.3
        )
        
        images_data.append({
            'id': log.id,
            'image_url': image_url,
            'created_at': log.created_at.isoformat(),
            'dominant_emotion': log.dominant_emotion,
            'emotion_ratio': log.dominant_emotion_ratio,
            'focus_level': log.focus_level,
            'fatigue_level': log.fatigue_level,
            'pose': log.pose or '',
            'wellbeing_score': wellbeing_score
        })
    
    if len(images_data) < 3:
        return Response({
            'error': '분석 가능한 이미지가 3개 미만입니다.',
            'images': []
        }, status=status.HTTP_200_OK)
    
    # LLM에게 이미지 데이터 제공하여 3개 선정
    system_prompt = """당신은 웰빙 상태 분석 전문가입니다. 사용자의 웰빙 이미지와 지표 데이터를 분석하여 다음 3가지 카테고리로 분류합니다:

1. Negative: 좋지 않은 상태 (sad/angry 감정, 낮은 집중도, 높은 피로도)
2. Normal: 중립 또는 보통 상태 (neutral 감정, 중간 수준의 지표)
3. Positive: 좋은 상태 (happy 감정, 높은 집중도, 낮은 피로도)

각 이미지의 ID와 지표 데이터를 분석하여, 각 카테고리에서 대표적인 이미지 1개씩 총 3개를 선정합니다.
응답은 반드시 JSON 형식으로만 제공하며, 다음과 같은 형식을 따릅니다:
{
  "negative": {"id": 이미지ID, "reason": "선정 이유"},
  "normal": {"id": 이미지ID, "reason": "선정 이유"},
  "positive": {"id": 이미지ID, "reason": "선정 이유"}
}"""
    
    user_prompt = f"""다음은 최근 하루 동안의 웰빙 이미지와 지표 데이터입니다:

{json.dumps(images_data, ensure_ascii=False, indent=2)}

위 이미지들 중에서 Negative, Normal, Positive 상태를 대표하는 이미지를 각각 1개씩 선정해주세요.
각 이미지의 지표 데이터(dominant_emotion, focus_level, fatigue_level, wellbeing_score)를 종합적으로 고려하여 선정하세요.

응답은 반드시 JSON 형식으로만 제공하세요."""
    
    try:
        llm_response = get_llm_response(user_prompt, system_prompt)
        
        # LLM 응답에서 JSON 추출
        import re
        json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)
        if json_match:
            selected_data = json.loads(json_match.group())
        else:
            # JSON 형식이 아니면 wellbeing_score를 기반으로 자동 선정
            sorted_images = sorted(images_data, key=lambda x: x['wellbeing_score'])
            selected_data = {
                "negative": {"id": sorted_images[0]['id'], "reason": "가장 낮은 웰빙 점수"},
                "normal": {"id": sorted_images[len(sorted_images)//2]['id'], "reason": "중간 웰빙 점수"},
                "positive": {"id": sorted_images[-1]['id'], "reason": "가장 높은 웰빙 점수"}
            }
        
        # 선정된 이미지 정보 반환
        selected_images = []
        image_dict = {img['id']: img for img in images_data}
        
        for category in ['negative', 'normal', 'positive']:
            if category in selected_data and 'id' in selected_data[category]:
                img_id = selected_data[category]['id']
                if img_id in image_dict:
                    img_data = image_dict[img_id]
                    selected_images.append({
                        'id': img_id,
                        'image_url': img_data['image_url'],
                        'category': category,
                        'reason': selected_data[category].get('reason', ''),
                        'created_at': img_data['created_at'],
                        'dominant_emotion': img_data['dominant_emotion'],
                        'focus_level': img_data['focus_level'],
                        'fatigue_level': img_data['fatigue_level']
                    })
        
        # 3개가 아니면 자동으로 보완
        if len(selected_images) < 3:
            sorted_images = sorted(images_data, key=lambda x: x['wellbeing_score'])
            categories_needed = ['negative', 'normal', 'positive']
            existing_categories = [img['category'] for img in selected_images]
            
            for category in categories_needed:
                if category not in existing_categories:
                    if category == 'negative':
                        img = sorted_images[0]
                    elif category == 'normal':
                        img = sorted_images[len(sorted_images)//2]
                    else:
                        img = sorted_images[-1]
                    
                    selected_images.append({
                        'id': img['id'],
                        'image_url': img['image_url'],
                        'category': category,
                        'reason': f'{category} 상태로 자동 선정',
                        'created_at': img['created_at'],
                        'dominant_emotion': img['dominant_emotion'],
                        'focus_level': img['focus_level'],
                        'fatigue_level': img['fatigue_level']
                    })
        
        return Response({
            'images': selected_images[:3],
            'total_images': len(images_data)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        import traceback
        return Response({
            'error': f'이미지 선정 오류: {str(e)}',
            'traceback': traceback.format_exc(),
            'images': []
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wellbeing_diagnosis(request):
    """
    사용자의 웰빙 상태를 종합적으로 진단하는 API.
    모든 촬영된 이미지와 웰빙 로그를 분석합니다.
    """
    user = request.user
    
    # 사용자 데이터 분석
    analysis_data = analyze_user_wellbeing(user)
    
    # 최근 포스트 이미지 정보
    recent_posts = Post.objects.filter(author=user).order_by('-created_date')[:10]
    post_summaries = []
    for post in recent_posts:
        post_summaries.append({
            "title": post.title,
            "text": post.text[:200] if post.text else "",
            "date": post.created_date.strftime("%Y-%m-%d %H:%M"),
            "has_image": bool(post.image)
        })
    
    # LLM 프롬프트 구성
    system_prompt = """당신은 전문 웰빙 상담사입니다. 사용자의 감정 데이터, 활동성 데이터, 그리고 촬영된 이미지 정보를 바탕으로 종합적인 웰빙 상태를 분석하고 상담하듯이 따뜻하고 공감적인 톤으로 답변해야 합니다.

답변 규칙:
1. 상담사처럼 공감하고 이해하는 태도로 답변
2. 한 메시지는 반드시 3줄을 넘지 않도록 간결하게 작성
3. 감정 상태, 생활 패턴, 전반적인 웰빙을 종합적으로 분석
4. 구체적이고 실용적인 조언 제공
5. 한국어로 자연스럽고 따뜻한 톤으로 대화

중요: 응답은 항상 5줄 이하로 작성하세요."""

    user_prompt = f"""다음은 사용자의 웰빙 데이터입니다:

**감정 분석:**
- 총 로그 수: {analysis_data['total_logs']}개
- 주요 감정: {analysis_data['dominant_emotion']}
- 감정 분포: {json.dumps(analysis_data['emotion_distribution'], ensure_ascii=False)}

**활동성 분석:**
- 평균 활동성: {analysis_data['avg_movement']}

**촬영 기록:**
- 총 포스트 수: {analysis_data['total_posts']}개
- 최근 포스트 요약:
{json.dumps(post_summaries, ensure_ascii=False, indent=2)}

이 데이터를 바탕으로 사용자의 현재 웰빙 상태를 종합적으로 진단하고, 감정 상태, 생활 패턴, 전반적인 웰빙에 대해 분석해주세요. 또한 구체적인 개선 제안도 함께 제공해주세요.

그리고 답변 메시지는 3줄을 넘지 않도록 간략하게 답변해줘. 그리고 편하게 대화하는 것처럼 반말로 답변해줘. 친근하게.
"""



    # LLM 호출
    diagnosis = get_llm_response(user_prompt, system_prompt)
    
    return Response({
        "analysis": diagnosis,
        "data_summary": analysis_data,
        "timestamp": timezone.now().isoformat()
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wellbeing_chat(request):
    """
    사용자와 웰빙 진단 에이전트 간의 대화 API.
    """
    user = request.user
    user_message = request.data.get('message', '')
    
    if not user_message:
        return Response({"error": "메시지가 필요합니다."}, status=status.HTTP_400_BAD_REQUEST)
    
    # 최근 웰빙 데이터 가져오기
    analysis_data = analyze_user_wellbeing(user)
    
    system_prompt = """당신은 전문 웰빙 상담사입니다. 사용자의 질문에 상담하듯이 공감하고 이해하는 태도로 답변해야 합니다.

답변 규칙:
1. 상담사처럼 공감적이고 따뜻한 톤으로 답변
2. 한 메시지는 반드시 5줄을 넘지 않도록 간결하게 작성
3. 웰빙 데이터를 참고하여 구체적이고 실용적인 조언 제공
4. 한국어로 자연스럽게 대화

중요: 응답은 항상 5줄 이하로 작성하세요."""

    user_prompt = f"""사용자가 다음과 같이 질문했습니다: "{user_message}"

현재 사용자의 웰빙 데이터:
- 주요 감정: {analysis_data.get('dominant_emotion', 'N/A')}
- 감정 분포: {json.dumps(analysis_data.get('emotion_distribution', {}), ensure_ascii=False)}
- 평균 활동성: {analysis_data.get('avg_movement', 0)}
- 총 로그 수: {analysis_data.get('total_logs', 0)}개

이 정보를 참고하여 사용자의 질문에 답변하고, 필요하다면 웰빙 개선을 위한 조언을 제공하세요."""

    # LLM 호출
    response = get_llm_response(user_prompt, system_prompt)
    
    return Response({
        "response": response,
        "timestamp": timezone.now().isoformat()
    }, status=status.HTTP_200_OK)