from django.urls import path
from app.views import AlgorithmsGetView

urlpatterns = [
    path('methods/', AlgorithmsGetView.as_view(), name='methods'),
]