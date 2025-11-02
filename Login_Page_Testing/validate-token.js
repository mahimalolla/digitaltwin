// netlify/functions/validate-token.js
const jwt = require('jsonwebtoken');

exports.handler = async (event, context) => {
  const token = event.headers.authorization?.replace('Bearer ', '');
  
  if (!token) {
    return {
      statusCode: 401,
      body: JSON.stringify({ error: 'No token provided' })
    };
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    return {
      statusCode: 200,
      body: JSON.stringify({
        valid: true,
        user: decoded
      })
    };
  } catch (error) {
    return {
      statusCode: 401,
      body: JSON.stringify({ 
        valid: false,
        error: 'Invalid or expired token' 
      })
    };
  }
};