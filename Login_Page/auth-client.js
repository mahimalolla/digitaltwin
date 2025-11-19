// public/js/auth-client.js

class AuthClient {
  constructor() {
    this.token = this.getTokenFromUrl() || this.getTokenFromStorage();
    this.user = null;
  }

  // Initiate OAuth flow
  login() {
    window.location.href = '/.netlify/functions/auth';
  }

  // Logout
  logout() {
    localStorage.removeItem('auth_token');
    this.token = null;
    this.user = null;
    window.location.href = '/index.html';
  }

  // Get token from URL params (after redirect)
  getTokenFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');
    if (token) {
      localStorage.setItem('auth_token', token);
      // Clean URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    return token;
  }

  // Get token from localStorage
  getTokenFromStorage() {
    return localStorage.getItem('auth_token');
  }

  // Validate token and get user info
  async validateSession() {
    if (!this.token) return false;

    try {
      const response = await fetch('/.netlify/functions/validate-token', {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });

      const data = await response.json();
      
      if (data.valid) {
        this.user = data.user;
        return true;
      } else {
        this.logout();
        return false;
      }
    } catch (error) {
      console.error('Session validation error:', error);
      return false;
    }
  }

  // Get current user
  getUser() {
    return this.user;
  }

  // Check if logged in
  isAuthenticated() {
    return !!this.token && !!this.user;
  }
}

// Initialize auth client
const auth = new AuthClient();

// Auto-validate on page load
document.addEventListener('DOMContentLoaded', async () => {
  const isValid = await auth.validateSession();
  
  if (window.location.pathname === '/dashboard.html' && !isValid) {
    // Redirect to login if trying to access protected page
    window.location.href = '/index.html';
  } else if (isValid && window.location.pathname === '/index.html') {
    // Redirect to dashboard if already logged in
    window.location.href = '/dashboard.html';
  }
  
  // Update UI based on auth state
  updateUIForAuthState(isValid);
});

function updateUIForAuthState(isAuthenticated) {
  const loginBtn = document.getElementById('loginBtn');
  const logoutBtn = document.getElementById('logoutBtn');
  const userInfo = document.getElementById('userInfo');
  
  if (isAuthenticated) {
    if (loginBtn) loginBtn.style.display = 'none';
    if (logoutBtn) logoutBtn.style.display = 'block';
    if (userInfo && auth.user) {
      userInfo.innerHTML = `
        <img src="${auth.user.picture}" alt="Profile" style="width:30px;border-radius:50%;">
        <span>${auth.user.name}</span>
      `;
    }
  } else {
    if (loginBtn) loginBtn.style.display = 'block';
    if (logoutBtn) logoutBtn.style.display = 'none';
    if (userInfo) userInfo.innerHTML = '';
  }
}