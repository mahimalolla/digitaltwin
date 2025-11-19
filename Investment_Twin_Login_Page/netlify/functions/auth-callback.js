// netlify/functions/auth-callback.js
const fetch = require('node-fetch');
const jwt = require('jsonwebtoken');

exports.handler = async (event, context) => {
  console.log('Auth callback function called');
  console.log('Query params:', event.queryStringParameters);

  // Get the authorization code and state
  const { code, state, error } = event.queryStringParameters || {};

  if (error) {
    console.error('OAuth error:', error);
    return {
      statusCode: 302,
      headers: {
        Location: '/?error=oauth_error&message=' + encodeURIComponent(error)
      },
      body: ''
    };
  }

  if (!code) {
    console.error('No authorization code received');
    return {
      statusCode: 302,
      headers: {
        Location: '/?error=no_code'
      },
      body: ''
    };
  }

  const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
  const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
  const JWT_SECRET = process.env.JWT_SECRET;

  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET || !JWT_SECRET) {
    console.error('Missing required environment variables');
    return {
      statusCode: 302,
      headers: {
        Location: '/?error=config_error'
      },
      body: ''
    };
  }

  const isLocal = event.headers.host?.includes('localhost') || 
                 event.headers.host?.includes('127.0.0.1');
  
  const REDIRECT_URI = isLocal 
    ? 'http://localhost:8888/auth/callback'
    : 'https://investmenttwin.netlify.app/auth/callback';

  try {
    console.log('Exchanging code for tokens...');
    
    // Exchange authorization code for tokens
    const tokenResponse = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        code: code,
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        redirect_uri: REDIRECT_URI,
        grant_type: 'authorization_code',
      }).toString(),
    });

    const tokenData = await tokenResponse.json();
    console.log('Token exchange response received');
    
    if (tokenData.error) {
      console.error('Token exchange error:', tokenData.error, tokenData.error_description);
      throw new Error(tokenData.error_description || tokenData.error);
    }

    // Get user information
    console.log('Fetching user information...');
    const userResponse = await fetch('https://www.googleapis.com/oauth2/v2/userinfo', {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
      },
    });

    const userInfo = await userResponse.json();
    console.log('User authenticated:', userInfo.email);

    const sessionToken = jwt.sign(
  {
    userId: userInfo.id,
    email: userInfo.email,
    name: userInfo.name,
    picture: userInfo.picture,
    exp: Math.floor(Date.now() / 1000) + (7 * 24 * 60 * 60),
  },
  JWT_SECRET,
  { algorithm: 'HS256' }
);

// Redirect to index with token in URL
return {
  statusCode: 302,
  headers: {
    Location: `/?token=${sessionToken}&success=true`,
    'Cache-Control': 'no-cache',
  },
  body: ''
};

    // Create a simple HTML response that sets the token and redirects
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Authentication Successful</title>
      </head>
      <body>
        <script>
          localStorage.setItem('auth_token', '${sessionToken}');
          window.location.href = '/dashboard.html';
        </script>
      </body>
      </html>
    `;

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'text/html',
      },
      body: html
    };

  } catch (error) {
    console.error('OAuth callback error:', error);
    return {
      statusCode: 302,
      headers: {
        Location: '/?error=auth_failed&message=' + encodeURIComponent(error.message)
      },
      body: ''
    };
  }
};