from django.urls import path
from .views import home, analyze_file, twitter_analysis, download_processed_csv, statistics_view, manage_dictionary

urlpatterns = [
    path('', home, name='home'),
    path('upload/', analyze_file, name='upload'),
    path('results/', analyze_file, name='results'),
    path('twitter/', twitter_analysis, name='twitter'),
    path('download/', download_processed_csv, name='download_csv'),
    path('statistics/', statistics_view, name='statistics'),
    path("manage-dictionary/", manage_dictionary, name="manage_dictionary"),
]
