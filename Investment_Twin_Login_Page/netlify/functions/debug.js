// netlify/functions/debug.js
exports.handler = async (event, context) => {
  // This is a debug endpoint to help troubleshoot OAuth setup
  // IMPORTANT: Disable or protect this in production!

  const isLocal = event.headers.host?.includes('localhost') || 
                 event.headers.host?.includes('127.0.0.1');

  // Only allow in development/local environment
  if (!isLocal && process.env.NODE_ENV === 'production') {
    return {
      statusCode: 404,
      body: JSON.stringify({ error: 'Not found' })
    };
  }

  // Collect debug information
  const debugInfo = {
    message: 'OAuth Debug Information',
    timestamp: new Date().toISOString(),
    environment: {
      isLocal: isLocal,
      nodeEnv: process.env.NODE_ENV || 'not set',
      host: event.headers.host,
    },
    env: {
      hasClientId: !!process.env.GOOGLE_CLIENT_ID,
      hasClientSecret: !!process.env.GOOGLE_CLIENT_SECRET,
      hasJwtSecret: !!process.env.JWT_SECRET,
      hasRedirectUri: !!process.env.REDIRECT_URI,
      redirectUri: process.env.REDIRECT_URI || 'not set',
      // Show partial client ID for verification (first 10 chars only)
      clientIdPreview: process.env.GOOGLE_CLIENT_ID 
        ? process.env.GOOGLE_CLIENT_ID.substring(0, 10) + '...' 
        : 'not set',
    },
    request: {
      httpMethod: event.httpMethod,
      path: event.path,
      headers: {
        host: event.headers.host,
        userAgent: event.headers['user-agent'],
        contentType: event.headers['content-type'],
        hasAuthorization: !!event.headers.authorization,
        hasCookie: !!event.headers.cookie,
      },
      queryParams: event.queryStringParameters || {},
      hasBody: !!event.body,
    },
    cookies: {},
    functionInfo: {
      functionName: context.functionName,
      functionVersion: context.functionVersion,
      remainingTime: context.getRemainingTimeInMillis ? context.getRemainingTimeInMillis() : 'N/A',
    },
    suggestions: []
  };

  // Parse cookies if present
  if (event.headers.cookie) {
    debugInfo.cookies = event.headers.cookie.split(';').reduce((acc, cookie) => {
      const [key, value] = cookie.trim().split('=');
      // Don't show full token values for security
      if (key === 'auth_token' || key === 'oauth_state') {
        acc[key] = value ? value.substring(0, 20) + '...' : 'empty';
      } else {
        acc[key] = value;
      }
      return acc;
    }, {});
  }

  // Add suggestions based on configuration
  if (!process.env.GOOGLE_CLIENT_ID) {
    debugInfo.suggestions.push('Set GOOGLE_CLIENT_ID environment variable');
  }
  if (!process.env.GOOGLE_CLIENT_SECRET) {
    debugInfo.suggestions.push('Set GOOGLE_CLIENT_SECRET environment variable');
  }
  if (!process.env.JWT_SECRET) {
    debugInfo.suggestions.push('Set JWT_SECRET environment variable');
  }
  if (!process.env.REDIRECT_URI && !isLocal) {
    debugInfo.suggestions.push('Set REDIRECT_URI environment variable for production');
  }

  // Test URLs for quick testing
  debugInfo.testUrls = {
    auth: `${isLocal ? 'http://localhost:8888' : 'https://investmenttwin.netlify.app'}/.netlify/functions/auth`,
    validateToken: `${isLocal ? 'http://localhost:8888' : 'https://investmenttwin.netlify.app'}/.netlify/functions/validate-token`,
    callback: `${isLocal ? 'http://localhost:8888' : 'https://investmenttwin.netlify.app'}/auth/callback`,
  };

  // Add OAuth flow status check
  if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET && process.env.JWT_SECRET) {
    debugInfo.oauthStatus = {
      configured: true,
      message: 'OAuth is properly configured',
      nextStep: 'Try visiting the auth URL to start OAuth flow'
    };
  } else {
    debugInfo.oauthStatus = {
      configured: false,
      message: 'OAuth is not fully configured',
      missingItems: debugInfo.suggestions
    };
  }

  // Return debug information
  return {
    statusCode: 200,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
    body: JSON.stringify(debugInfo, null, 2)
  };
};