from django import forms
from django.contrib.auth.models import User

class VideoUploadForm(forms.Form):
    upload_video_file = forms.FileField(label="Select Video", required=True, widget=forms.FileInput(attrs={"accept": "video/*"}))
    sequence_length = forms.IntegerField(label="Sequence Length", required=True)

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    middle_name = forms.CharField(required=False, max_length=30)
    phone_no = forms.CharField(required=False, max_length=15)

    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name', 'password']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])
        if commit:
            user.save()
            # Create or update the UserProfile with additional fields
            UserProfile.objects.create(user=user, middle_name=self.cleaned_data['middle_name'], phone_no=self.cleaned_data['phone_no'])
        return user

class UserLoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)
