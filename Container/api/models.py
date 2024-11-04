from django.db import models
from django.contrib.auth.models import User

from django import forms



class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True)

    def __str__(self):
        return self.user.username

class Skin(models.Model):
    name = models.CharField(max_length=50)
    skin_Main_Img = models.ImageField(upload_to='images/')
    
    def __str__(self):
        return self.name  
    
class SkinForm(forms.Form):
    skin_Main_Img = forms.ImageField()
    name = forms.CharField(max_length=255, required=False)
    
class Upload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    upload_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default='pending')
    name = models.CharField(max_length=255, blank=True, null=True)  

    def __str__(self):
        return f"{self.file_name} uploaded by {self.user.username}"