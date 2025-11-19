// netlify/functions/auth.js
exports.handler = async (event, context) => {
  const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
  
  if (!GOOGLE_CLIENT_ID) {
    return {
      statusCode: 500,
      body: JSON.stringify({ 
        error: 'OAuth configuration error', 
        message: 'Google Client ID not configured' 
      })
    };
  }

  const isLocal = event.headers.host?.includes('localhost') || 
                 event.headers.host?.includes('127.0.0.1');
  
  // Use the /auth/callback URL (not the function URL directly)
  const REDIRECT_URI = isLocal 
    ? 'http://localhost:8888/auth/callback'  // This will be redirected by netlify.toml
    : 'https://investmenttwin.netlify.app/auth/callback';
  
  const generateRandomState = () => {
    return Math.random().toString(36).substring(2, 15) + 
           Math.random().toString(36).substring(2, 15);
  };

  const state = generateRandomState();
  
  const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
    `client_id=${GOOGLE_CLIENT_ID}&` +
    `redirect_uri=${encodeURIComponent(REDIRECT_URI)}&` +
    `response_type=code&` +
    `scope=${encodeURIComponent('email profile openid')}&` +
    `access_type=offline&` +
    `state=${state}&` +
    `prompt=consent`;

  console.log('Redirecting to Google OAuth...');
  console.log('Redirect URI being used:', REDIRECT_URI);

  return {
    statusCode: 302,
    headers: {
      Location: authUrl,
      'Cache-Control': 'no-cache'
    },
    body: ''
  };
};