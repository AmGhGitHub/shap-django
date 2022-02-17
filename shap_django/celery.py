import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", 'shap_django.settings')

celery_app = Celery('shap_django')
celery_app.config_from_object('django.conf:settings', namespace='CELERY')
celery_app.autodiscover_tasks()