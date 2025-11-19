// netlify/functions/auth.js
exports.handler = async (event, context) => {
  const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
  const REDIRECT_URI = process.env.REDIRECT_URI;
  
  const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
    `client_id=${GOOGLE_CLIENT_ID}&` +
    `redirect_uri=${encodeURIComponent(REDIRECT_URI)}&` +
    `response_type=code&` +
    `scope=email%20profile%20openid&` +
    `access_type=offline&` +
    `state=${generateRandomState()}&` +
    `prompt=consent`;

  return {
    statusCode: 302,
    headers: {
      Location: authUrl,
      'Cache-Control': 'no-cache'
    }
  };
};

function generateRandomState() {
  return Math.random().toString(36).substring(2, 15) + 
         Math.random().toString(36).substring(2, 15);
}