"""
Django 관리자 계정 확인 및 비밀번호 재설정 스크립트
"""

import os
import sys
import io

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import django

# Django 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PhotoBlogServer.settings")
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

def check_admin_accounts():
    """모든 관리자 계정 확인"""
    print("="*60)
    print("  Django 관리자 계정 확인")
    print("="*60)
    print()
    
    # 모든 사용자 조회
    users = User.objects.all()
    
    if not users.exists():
        print("❌ 등록된 사용자가 없습니다.")
        print("\n관리자 계정을 생성하려면:")
        print("  python manage.py createsuperuser")
        return
    
    print(f"총 {users.count()}명의 사용자가 등록되어 있습니다.\n")
    
    # 관리자 계정 목록
    admin_users = users.filter(is_superuser=True)
    staff_users = users.filter(is_staff=True)
    
    if admin_users.exists():
        print("="*60)
        print("  관리자 계정 (Superuser)")
        print("="*60)
        for user in admin_users:
            print(f"  ID: {user.id}")
            print(f"  Username: {user.username}")
            print(f"  Email: {user.email}")
            print(f"  Is Staff: {user.is_staff}")
            print(f"  Is Superuser: {user.is_superuser}")
            print(f"  Last Login: {user.last_login}")
            print(f"  Date Joined: {user.date_joined}")
            print()
    
    if staff_users.exists() and not admin_users.exists():
        print("="*60)
        print("  스태프 계정 (Staff)")
        print("="*60)
        for user in staff_users:
            print(f"  ID: {user.id}")
            print(f"  Username: {user.username}")
            print(f"  Email: {user.email}")
            print()
    
    print("="*60)
    print("  모든 사용자 목록")
    print("="*60)
    for user in users:
        admin_status = "관리자" if user.is_superuser else ("스태프" if user.is_staff else "일반")
        print(f"  [{admin_status}] {user.username} ({user.email})")
    
    print()
    print("="*60)
    print("  비밀번호 확인/재설정 방법")
    print("="*60)
    print()
    print("[주의] Django는 보안상 비밀번호를 해시로 저장하므로")
    print("       원본 비밀번호를 확인할 수 없습니다.")
    print()
    print("비밀번호를 재설정하려면:")
    print("  1. python manage.py changepassword <username>")
    print("  2. 또는 아래 스크립트 사용")
    print("     python check_admin.py admin newpassword")
    print()

def reset_password(username, new_password):
    """관리자 비밀번호 재설정"""
    try:
        user = User.objects.get(username=username)
        user.set_password(new_password)
        user.save()
        print(f"✅ 비밀번호가 성공적으로 변경되었습니다: {username}")
        return True
    except User.DoesNotExist:
        print(f"❌ 사용자를 찾을 수 없습니다: {username}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 비밀번호 재설정 모드
        if len(sys.argv) == 3:
            username = sys.argv[1]
            new_password = sys.argv[2]
            reset_password(username, new_password)
        else:
            print("사용법: python check_admin.py <username> <new_password>")
            print("예시: python check_admin.py admin newpassword123")
    else:
        # 계정 확인 모드
        check_admin_accounts()

