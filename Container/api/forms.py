from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Skin

class CustomUserCreationForm(UserCreationForm):
    
    
    first_name = forms.CharField(
        max_length=30, 
        required=True, 
        help_text='Enter your first name.'
    )
    last_name = forms.CharField(
        max_length=30, 
        required=True, 
        help_text='Enter your last name.'
    )
    email = forms.EmailField(
        max_length=254, 
        help_text='Enter a valid email address.'
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput, 
        help_text='Password must be at least 8 characters long.'
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput, 
        help_text='Enter the same password as before, for verification.'
    )

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')

class SkinForm(forms.ModelForm):
    
   
    class Meta:
        model = Skin
        fields = ['name', 'skin_Main_Img']
