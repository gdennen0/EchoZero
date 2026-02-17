"""
Simple Authentication Service

Handles user login/registration only.
All billing/payment handled by Stripe Billing service.
"""
from typing import Optional, Dict, Any
import httpx
import os
from datetime import datetime

from src.utils.message import Log


class AuthService:
    """
    Simple authentication service.
    
    Responsibilities:
    - User login/registration
    - Session management
    - Link to Stripe Customer ID
    
    Does NOT handle:
    - Payment processing (Stripe handles this)
    - Billing (Stripe handles this)
    - Invoicing (Stripe handles this)
    """
    
    def __init__(self, backend_url: Optional[str] = None):
        """
        Initialize auth service.
        
        Args:
            backend_url: URL of backend service (for login/register)
        """
        self._backend_url = backend_url or os.getenv("ECHOZERO_BACKEND_URL", "http://localhost:8000")
        self.current_user: Optional[Dict[str, Any]] = None
        self._access_token: Optional[str] = None
        
        Log.info(f"AuthService: Initialized (backend: {self._backend_url})")
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Login user.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            User data including Stripe Customer ID
            
        Raises:
            Exception: If login fails
        """
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self._backend_url}/auth/login",
                    json={"email": email, "password": password},
                    timeout=10.0
                )
                response.raise_for_status()
                user_data = response.json()
            
            self.current_user = user_data
            self._access_token = user_data.get("access_token")
            
            Log.info(f"AuthService: User logged in: {user_data.get('email')}")
            return user_data
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise Exception("Invalid email or password")
            raise Exception(f"Login failed: {e.response.text}")
        except Exception as e:
            Log.error(f"AuthService: Login error: {e}")
            raise
    
    def register(self, email: str, password: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Register new user.
        
        Backend automatically:
        - Creates user account
        - Creates Stripe Customer
        - Sets up bi-weekly billing subscription
        
        Args:
            email: User email
            password: User password
            name: Optional user name
            
        Returns:
            User data including Stripe Customer ID
            
        Raises:
            Exception: If registration fails
        """
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self._backend_url}/auth/register",
                    json={
                        "email": email,
                        "password": password,
                        "name": name
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                user_data = response.json()
            
            self.current_user = user_data
            self._access_token = user_data.get("access_token")
            
            Log.info(f"AuthService: User registered: {user_data.get('email')}")
            return user_data
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                raise Exception("Email already registered")
            raise Exception(f"Registration failed: {e.response.text}")
        except Exception as e:
            Log.error(f"AuthService: Registration error: {e}")
            raise
    
    def logout(self) -> None:
        """Logout current user"""
        if self.current_user:
            email = self.current_user.get("email", "unknown")
            Log.info(f"AuthService: User logged out: {email}")
        
        self.current_user = None
        self._access_token = None
    
    def is_logged_in(self) -> bool:
        """Check if user is logged in"""
        return self.current_user is not None
    
    def get_user_id(self) -> Optional[str]:
        """Get current user ID"""
        if self.current_user:
            return self.current_user.get("user_id")
        return None
    
    def get_stripe_customer_id(self) -> Optional[str]:
        """
        Get Stripe Customer ID for current user.
        
        Used for:
        - Reporting usage to Stripe Billing
        - Accessing billing portal
        """
        if self.current_user:
            return self.current_user.get("stripe_customer_id")
        return None
    
    def get_access_token(self) -> Optional[str]:
        """Get access token for API calls"""
        return self._access_token
    
    def get_user_email(self) -> Optional[str]:
        """Get current user email"""
        if self.current_user:
            return self.current_user.get("email")
        return None
    
    def refresh_session(self) -> bool:
        """
        Refresh user session (validate token still valid).
        
        Returns:
            True if session valid, False otherwise
        """
        if not self._access_token:
            return False
        
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self._backend_url}/auth/me",
                    headers={"Authorization": f"Bearer {self._access_token}"},
                    timeout=10.0
                )
                response.raise_for_status()
                user_data = response.json()
            
            self.current_user = user_data
            return True
        
        except Exception as e:
            Log.warning(f"AuthService: Session refresh failed: {e}")
            self.logout()
            return False
