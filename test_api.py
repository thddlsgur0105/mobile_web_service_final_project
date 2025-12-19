"""
RESTful API 테스트 스크립트
모든 API 엔드포인트를 테스트하고 결과를 출력합니다.
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_get_posts():
    """Post 목록 조회 테스트"""
    print_section("1. GET /api_root/Post/ - Post 목록 조회")
    try:
        response = requests.get(f"{BASE_URL}/api_root/Post/")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 성공! 총 {len(data.get('results', data))}개의 포스트")
            if data.get('results'):
                print("\n최근 포스트:")
                for post in data.get('results', data)[:3]:
                    print(f"  - ID: {post.get('id')}, 제목: {post.get('title')}")
        else:
            print(f"❌ 실패: {response.text}")
    except Exception as e:
        print(f"❌ 오류: {e}")

def test_post_wellbeing_log():
    """WellbeingLog 생성 테스트"""
    print_section("2. POST /api_root/WellbeingLog/ - WellbeingLog 생성")
    try:
        data = {
            "dominant_emotion": "happy",
            "dominant_emotion_ratio": 0.75,
            "emotion_counts": {"happy": 10, "sad": 2, "neutral": 5},
            "avg_movement": 15.5
        }
        response = requests.post(
            f"{BASE_URL}/api_root/WellbeingLog/",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"✅ 성공! 생성된 로그 ID: {result.get('id')}")
            print(f"   감정: {result.get('dominant_emotion')}")
            print(f"   활동성: {result.get('avg_movement')}")
        else:
            print(f"❌ 실패: {response.text}")
    except Exception as e:
        print(f"❌ 오류: {e}")

def test_get_wellbeing_logs():
    """WellbeingLog 목록 조회 테스트"""
    print_section("3. GET /api_root/WellbeingLog/ - WellbeingLog 목록 조회")
    try:
        response = requests.get(f"{BASE_URL}/api_root/WellbeingLog/")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logs = data.get('results', data) if isinstance(data, dict) else data
            print(f"✅ 성공! 총 {len(logs)}개의 로그")
            if logs:
                print("\n최근 로그:")
                for log in logs[:3]:
                    print(f"  - ID: {log.get('id')}, 감정: {log.get('dominant_emotion')}, "
                          f"활동성: {log.get('avg_movement')}")
        else:
            print(f"❌ 실패: {response.text}")
    except Exception as e:
        print(f"❌ 오류: {e}")

def test_get_wellbeing_summary():
    """WellbeingLog 요약 조회 테스트"""
    print_section("4. GET /api_root/WellbeingLog/summary/ - WellbeingLog 요약")
    try:
        response = requests.get(f"{BASE_URL}/api_root/WellbeingLog/summary/")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ 성공!")
            print("\n오늘 요약:")
            today = data.get('today', {})
            print(f"  - 로그 수: {today.get('count')}")
            print(f"  - 주요 감정: {today.get('dominant_emotion')}")
            print(f"  - 평균 활동성: {today.get('avg_movement')}")
            
            print("\n최근 7일 요약:")
            week = data.get('last_7_days', {})
            print(f"  - 로그 수: {week.get('count')}")
            print(f"  - 주요 감정: {week.get('dominant_emotion')}")
            print(f"  - 평균 활동성: {week.get('avg_movement')}")
        else:
            print(f"❌ 실패: {response.text}")
    except Exception as e:
        print(f"❌ 오류: {e}")

def test_get_single_post():
    """특정 Post 조회 테스트"""
    print_section("5. GET /api_root/Post/{id}/ - 특정 Post 조회")
    try:
        # 먼저 목록을 가져와서 ID 확인
        response = requests.get(f"{BASE_URL}/api_root/Post/")
        if response.status_code == 200:
            data = response.json()
            posts = data.get('results', data) if isinstance(data, dict) else data
            if posts:
                post_id = posts[0].get('id')
                response = requests.get(f"{BASE_URL}/api_root/Post/{post_id}/")
                print(f"Status Code: {response.status_code}")
                if response.status_code == 200:
                    post = response.json()
                    print(f"✅ 성공! Post ID: {post_id}")
                    print(f"   제목: {post.get('title')}")
                    print(f"   내용: {post.get('text')[:50]}...")
                else:
                    print(f"❌ 실패: {response.text}")
            else:
                print("⚠️ 조회할 포스트가 없습니다.")
        else:
            print(f"❌ Post 목록 조회 실패: {response.text}")
    except Exception as e:
        print(f"❌ 오류: {e}")

def main():
    """모든 API 테스트 실행"""
    print("\n" + "="*60)
    print("  RESTful API 테스트 시작")
    print("="*60)
    print(f"\n서버 주소: {BASE_URL}")
    print("Django 서버가 실행 중인지 확인하세요!\n")
    
    # 서버 연결 확인
    try:
        response = requests.get(f"{BASE_URL}/api_root/", timeout=2)
        print("✅ 서버 연결 성공!\n")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        print("\nDjango 서버를 먼저 실행하세요:")
        print("  python manage.py runserver")
        return
    
    # 각 API 테스트 실행
    test_get_posts()
    test_post_wellbeing_log()
    test_get_wellbeing_logs()
    test_get_wellbeing_summary()
    test_get_single_post()
    
    print("\n" + "="*60)
    print("  테스트 완료!")
    print("="*60)
    print("\n브라우저에서도 확인 가능:")
    print(f"  - Post 목록: {BASE_URL}/api_root/Post/")
    print(f"  - WellbeingLog 목록: {BASE_URL}/api_root/WellbeingLog/")
    print(f"  - WellbeingLog 요약: {BASE_URL}/api_root/WellbeingLog/summary/")
    print()

if __name__ == "__main__":
    main()

