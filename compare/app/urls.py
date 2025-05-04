from django.urls import path
from app.views import AlgorithmsPostView, ConvertPdfView

urlpatterns = [
    path('methods/', AlgorithmsPostView.as_view(), name='methods'),
    path("convert-pdf/", ConvertPdfView.as_view(), name="convert-pdf"),
]