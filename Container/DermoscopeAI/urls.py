from django.contrib import admin
from django.urls import path
from api.views import (
    login_view, register, home, about, read, skin_image_view, success,
    profile_view, privacyPolicy, termsOfService, view_results, 
    view_cumulative_results, check_results, logout_view
)
from django.contrib.auth.views import LogoutView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('about/', about, name='about'),
    path('login/', login_view, name='login'),
    path('register/', register, name='register'),
    path('read/', read, name='read'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('upload/', skin_image_view, name='upload'),
    path('success/', success, name='success'),
    path('profile/', profile_view, name='profile'),
    path('privacyPolicy/', privacyPolicy, name='privacyPolicy'),
    path('termsOfService/', termsOfService, name='termsOfService'),
    path('results/<str:file_name>/', view_results, name='view_results'),
    path('view_cumulative_results/', view_cumulative_results, name='view_cumulative_results'),
    path('check_results/', check_results, name='check_results'),
    path('logout/', logout_view, name='logout'), 
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
