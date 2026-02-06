
import { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { upstoxService } from '../../services/upstoxService';

const AuthCallback = () => {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const code = searchParams.get('code');

    useEffect(() => {
        if (code) {
            // In a real app, we would send this code to our backend to exchange for a token
            // For now, let's assume our backend exchange endpoint is exposed 
            // and we can fetch the token from there.

            const exchangeToken = async () => {
                try {
                    // Call our backend to exchange code for token
                    const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/auth/upstox/callback?code=${code}`);
                    if (response.ok) {
                        const data = await response.json();
                        if (data.access_token) {
                            upstoxService.setAccessToken(data.access_token);
                            localStorage.setItem('upstox_access_token', data.access_token);
                            navigate('/');
                        }
                    } else {
                        console.error('Failed to exchange token');
                        navigate('/?error=auth_failed');
                    }
                } catch (error) {
                    console.error('Error exchanging token:', error);
                    navigate('/?error=auth_error');
                }
            };

            exchangeToken();
        }
    }, [code, navigate]);

    return (
        <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh',
            backgroundColor: '#0f172a',
            color: 'white'
        }}>
            <h2>Authenticating with Upstox...</h2>
        </div>
    );
};

export default AuthCallback;
