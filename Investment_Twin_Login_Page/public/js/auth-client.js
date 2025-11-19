// public/js/auth-client.js
class AuthClient {
    constructor() {
        this.token = this.getTokenFromUrl() || this.getTokenFromStorage();
        this.user = null;
        this.baseUrl = window.location.hostname === 'localhost' 
            ? 'http://localhost:8888' 
            : 'https://investmenttwin.netlify.app';
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

    // Logout
    logout() {
        localStorage.removeItem('auth_token');
        this.token = null;
        this.user = null;
        window.location.href = '/';
    }
}