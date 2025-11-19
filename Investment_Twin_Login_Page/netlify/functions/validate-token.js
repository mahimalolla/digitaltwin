javascript// netlify/functions/validate-token.js
const jwt = require('jsonwebtoken');

exports.handler = async (event, context) => {
  // Set CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  };

  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  // Only accept GET or POST requests
  if (event.httpMethod !== 'GET' && event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    // Get token from Authorization header or cookie
    let token = null;

    // Check Authorization header first
    const authHeader = event.headers.authorization || event.headers.Authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      token = authHeader.substring(7);
    }
    
    // If no Authorization header, check cookies
    if (!token && event.headers.cookie) {
      const cookies = event.headers.cookie.split(';').reduce((acc, cookie) => {
        const [key, value] = cookie.trim().split('=');
        acc[key] = value;
        return acc;
      }, {});
      token = cookies.auth_token;
    }

    // If still no token, check query parameters (less secure, but useful for testing)
    if (!token && event.queryStringParameters?.token) {
      token = event.queryStringParameters.token;
    }

    // No token found
    if (!token) {
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ 
          valid: false, 
          error: 'No token provided' 
        })
      };
    }

    // Get JWT secret
    const JWT_SECRET = process.env.JWT_SECRET;
    if (!JWT_SECRET) {
      console.error('JWT_SECRET not configured');
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ 
          valid: false, 
          error: 'Server configuration error' 
        })
      };
    }

    // Verify and decode token
    const decoded = jwt.verify(token, JWT_SECRET, {
      algorithms: ['HS256'],
      issuer: 'warren-buffett-digital-twin'
    });

    // Check if token is expired (jwt.verify handles this, but we can double-check)
    if (decoded.exp && decoded.exp * 1000 < Date.now()) {
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ 
          valid: false, 
          error: 'Token expired' 
        })
      };
    }

    // Token is valid
    console.log('Token validated for user:', decoded.email);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        valid: true,
        user: {
          userId: decoded.userId,
          email: decoded.email,
          name: decoded.name,
          picture: decoded.picture,
          verified_email: decoded.verified_email,
        },
        expires: decoded.exp ? new Date(decoded.exp * 1000).toISOString() : null
      })
    };

  } catch (error) {
    console.error('Token validation error:', error.message);
    
    // Handle specific JWT errors
    let errorMessage = 'Invalid token';
    let statusCode = 401;

    if (error.name === 'TokenExpiredError') {
      errorMessage = 'Token expired';
    } else if (error.name === 'JsonWebTokenError') {
      errorMessage = 'Invalid token format';
    } else if (error.name === 'NotBeforeError') {
      errorMessage = 'Token not yet valid';
    } else {
      statusCode = 500;
      errorMessage = 'Token validation failed';
    }

    return {
      statusCode,
      headers,
      body: JSON.stringify({ 
        valid: false, 
        error: errorMessage,
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      })
    };
  }
};