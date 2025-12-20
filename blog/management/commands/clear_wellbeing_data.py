"""
WellbeingLog 데이터를 모두 삭제하는 Django 관리 명령.

사용법:
    python manage.py clear_wellbeing_data
    python manage.py clear_wellbeing_data --user username  # 특정 사용자만 삭제
"""

from django.core.management.base import BaseCommand
from blog.models import WellbeingLog
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = 'WellbeingLog 데이터를 모두 삭제합니다.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            help='특정 사용자의 데이터만 삭제 (사용자명 지정)',
        )

    def handle(self, *args, **options):
        username = options.get('user')
        
        if username:
            try:
                user = User.objects.get(username=username)
                count = WellbeingLog.objects.filter(user=user).count()
                WellbeingLog.objects.filter(user=user).delete()
                self.stdout.write(
                    self.style.SUCCESS(
                        f'✅ {username} 사용자의 WellbeingLog 데이터 {count}개를 삭제했습니다.'
                    )
                )
            except User.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'❌ 사용자 "{username}"를 찾을 수 없습니다.')
                )
        else:
            count = WellbeingLog.objects.count()
            WellbeingLog.objects.all().delete()
            self.stdout.write(
                self.style.SUCCESS(
                    f'✅ 모든 WellbeingLog 데이터 {count}개를 삭제했습니다.'
                )
            )

