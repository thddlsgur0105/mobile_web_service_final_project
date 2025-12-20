"""
Post 데이터와 이미지 파일을 모두 삭제하는 Django 관리 명령.

사용법:
    python manage.py clear_posts
    python manage.py clear_posts --user username  # 특정 사용자만 삭제
"""

import os
from django.core.management.base import BaseCommand
from django.conf import settings
from blog.models import Post
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = 'Post 데이터와 이미지 파일을 모두 삭제합니다.'

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
                posts = Post.objects.filter(author=user)
                count = posts.count()
                
                # 이미지 파일 삭제
                deleted_files = 0
                for post in posts:
                    if post.image and post.image.name:
                        image_path = os.path.join(settings.MEDIA_ROOT, post.image.name)
                        if os.path.exists(image_path):
                            try:
                                os.remove(image_path)
                                deleted_files += 1
                            except Exception as e:
                                self.stdout.write(
                                    self.style.WARNING(f'⚠️ 이미지 파일 삭제 실패: {image_path} - {e}')
                                )
                
                # Post 레코드 삭제
                posts.delete()
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'✅ {username} 사용자의 Post 데이터 {count}개와 이미지 파일 {deleted_files}개를 삭제했습니다.'
                    )
                )
            except User.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'❌ 사용자 "{username}"를 찾을 수 없습니다.')
                )
        else:
            posts = Post.objects.all()
            count = posts.count()
            
            # 이미지 파일 삭제
            deleted_files = 0
            for post in posts:
                if post.image and post.image.name:
                    image_path = os.path.join(settings.MEDIA_ROOT, post.image.name)
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            deleted_files += 1
                        except Exception as e:
                            self.stdout.write(
                                self.style.WARNING(f'⚠️ 이미지 파일 삭제 실패: {image_path} - {e}')
                            )
            
            # Post 레코드 삭제
            posts.delete()
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'✅ 모든 Post 데이터 {count}개와 이미지 파일 {deleted_files}개를 삭제했습니다.'
                )
            )

