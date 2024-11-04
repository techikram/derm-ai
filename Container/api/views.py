from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from .forms import CustomUserCreationForm, SkinForm
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import subprocess
import logging
import csv
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.urls import reverse
from .models import Upload
from datetime import timedelta

logger = logging.getLogger(__name__)


def logout_view(request):
    auth_logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')

def view_results(request, file_name):
    error = None
    results = None
    user_results_dir = os.path.join('/home/falcon/student2/api/results', str(request.user.id))
    csv_filename = f"{file_name}_results.csv"
    csv_path = os.path.join(user_results_dir, csv_filename)

    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                results = list(reader)
        except Exception as e:
            error = f'Error reading CSV file: {e}'
    else:
        error = 'No CSV file found for this image.'

    context = {
        'file_name': file_name,
        'results': results,
        'error': error,
    }
    return render(request, 'view_results.html', context)

def view_cumulative_results(request):
    error = None
    cumulative_results = None
    user_results_dir = os.path.join('/home/falcon/student2/api/results', str(request.user.id))
    cumulative_csv_path = os.path.join(user_results_dir, "cumulative_results.csv")

    if os.path.exists(cumulative_csv_path):
        try:
            with open(cumulative_csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                cumulative_results = list(reader)
        except Exception as e:
            error = f'Error reading cumulative CSV file: {e}'
    else:
        error = 'No cumulative CSV file found.'

    context = {
        'cumulative_results': cumulative_results,
        'error': error,
    }
    return render(request, 'view_cumulative_results.html', context)

@login_required
def skin_image_view(request):
    if request.method == 'POST':
        form = SkinForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['skin_Main_Img']
            name = form.cleaned_data.get('name', '')  
            user = request.user
            user_data_path = os.path.join('/home/falcon/student2/api/img', str(user.id))
            save_path = os.path.join(user_data_path, file.name)

            if not os.path.exists(user_data_path):
                os.makedirs(user_data_path)

            with open(save_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            upload_record = Upload(user=user, file_name=file.name, name=name) 
            upload_record.save()

            try:
                result = subprocess.run(
                    ['python3', '/home/falcon/student2/api/test.py', str(user.id)],
                    check=True,
                    capture_output=True,
                    text=True,
                    env={
                        'VIRTUAL_ENV': '/home/falcon/student2/api/Container/venv',
                        'PATH': '/home/falcon/student2/api/Container/venv/bin:' + os.environ['PATH']
                    }
                )
                output = result.stdout
                error = result.stderr

                user_results_dir = os.path.join('/home/falcon/student2/api/results', str(user.id))
                csv_files = [f for f in os.listdir(user_results_dir) if f.endswith('.csv')]
                if csv_files:
                    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(user_results_dir, x)), reverse=True)
                    latest_csv = os.path.join(user_results_dir, csv_files[0])

                    results = []
                    with open(latest_csv, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            results.append(row)
                    
                    upload_record.status = 'success' 
                else:
                    results = []
                    upload_record.status = 'no_results'  
                upload_record.save()

                if os.path.exists(save_path):
                    os.remove(save_path)

                file_name_without_extension, _ = os.path.splitext(file.name)
                request.session['file_name'] = file_name_without_extension

                return redirect(reverse('view_results', args=[file_name_without_extension]))

            except subprocess.CalledProcessError as e:
                output = e.stdout
                error = e.stderr
                upload_record.status = 'error'  
                upload_record.save()
                if os.path.exists(save_path):
                    os.remove(save_path)
                return render(request, 'error.html', {'error': error, 'output': output})

    else:
        form = SkinForm()

    return render(request, 'upload.html', {'form': form})

@login_required
def profile_view(request):
    user = request.user
    uploads = Upload.objects.filter(user=user).order_by('-upload_date')
    results = {}
    for upload in uploads:
        upload.upload_date += timedelta(hours=2)
        user_results_dir = os.path.join('/home/falcon/student2/api/results', str(user.id))
        csv_files = [f for f in os.listdir(user_results_dir) if f.endswith('.csv')]
        if csv_files:
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(user_results_dir, x)), reverse=True)
            latest_csv = os.path.join(user_results_dir, csv_files[0])
            result_list = []
            with open(latest_csv, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    result_list.append(row)
            results[upload.id] = result_list
        else:
            results[upload.id] = []

    context = {
        'username': user.username,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'uploads': uploads,
        'results': results,
    }
    return render(request, 'profile.html', context)

@login_required
def check_results(request):
    user = request.user
    file_name = request.session.get('file_name')
    
    logger.debug(f"File name from session: {file_name}")

    if not file_name:
        logger.error("No file name found in session")
        return JsonResponse({'status': 'error', 'message': 'No file name found in session'})
    
    try:
        upload_record = Upload.objects.get(user=user, file_name=file_name)
    except Upload.DoesNotExist:
        logger.error(f"Upload record does not exist for user {user.id} and file name {file_name}")
        return JsonResponse({'status': 'error', 'message': 'Upload record does not exist'})
    
    
    if upload_record.status == 'success':
        return JsonResponse({'status': 'success'})
    elif upload_record.status == 'error':
        return JsonResponse({'status': 'error', 'message': 'An error occurred with the upload'})
    else:
        return JsonResponse({'status': 'pending'})

def success(request):
    return HttpResponse('Successfully uploaded')

def IndexView(request):
    return HttpResponse('<h1> Hi world  </h1>')

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)  
            messages.success(request, f'Successful registration: Welcome, {user.username}!')
            return redirect('upload')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            user = form.get_user()
            auth_login(request, user)  
            messages.success(request, f'Welcome, {username}!')
            return redirect('upload')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def home(request):
    user = request.user
    context = {
        'user': user
    }
    return render(request, 'home.html', context)

def about(request):
    return render(request, 'about.html')

def read(request):
    return render(request, 'read.html')

def upload(request):
    return render(request, 'upload.html')

def privacyPolicy(request):
    return render(request, 'privacyPolicy.html')

def termsOfService(request):
    return render(request, 'termsOfService.html')
