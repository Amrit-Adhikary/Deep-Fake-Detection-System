from django.db import models
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser, PermissionsMixin

# User role choices
USER_ROLES = (
    ("Admin", "Admin"),
    ("User", "User"),
)

class UserManager(BaseUserManager):
    def create_user(self, email, first_name, last_name, middle_name, phone_no, role, password=None):
        """
        Creates and saves a User with the given email, first_name, last_name, middle_name, 
        phone_no, role, and password.
        """
        if not email:
            raise ValueError('Users must have an email address')
        if not first_name:
            raise ValueError('Users must provide a first name')
        if not middle_name:
            raise ValueError('Users must provide a middle name')
        if not last_name:
            raise ValueError('Users must provide a last name')
        if not role:
            raise ValueError('Users must assign a role')

        user = self.model(
            email=self.normalize_email(email),
            first_name=first_name,
            last_name=last_name,
            middle_name=middle_name,
            phone_no=phone_no,
            role=role
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, first_name, middle_name, last_name, phone_no, password=None, role='Admin'):
        """
        Creates and saves a superuser with the given first_name, middle_name, last_name, 
        email, role, phone_no, and password.
        """
        user = self.create_user(
            email=email,
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            phone_no=phone_no,
            password=password,
            role=role  # Default role is Admin if not provided
        )
        user.is_admin = True
        user.is_superuser = True  # Ensure they have all permissions
        user.is_staff = True  # Set staff to True for superusers
        user.save(using=self._db)
        return user

class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )
    first_name = models.CharField(max_length=128, null=True, blank=True)
    middle_name = models.CharField(max_length=128, null=True, blank=True)
    last_name = models.CharField(max_length=128, null=True, blank=True)
    phone_no = models.CharField(max_length=10)
    role = models.CharField(choices=USER_ROLES, max_length=121, default='User')
    is_email_verified = models.BooleanField(default=False)

    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)  # Default value is False, set True for staff users
    is_superuser = models.BooleanField(default=False)  # Add this field to mark a superuser
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'middle_name', 'last_name', 'phone_no']

    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        """
        Check if the user has specific permissions.
        Simplified: returns True for all users.
        """
        return True

    def has_module_perms(self, app_label):
        """
        Check if the user has permissions to view the app `app_label`.
        Simplified: returns True for all users.
        """
        return True

    # is_staff is automatically handled by is_admin field
    # No need for the @property decorator here

