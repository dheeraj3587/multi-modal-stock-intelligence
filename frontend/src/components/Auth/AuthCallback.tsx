
import { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

const AuthCallback = () => {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const code = searchParams.get('code');

    useEffect(() => {
        if (code) {
            const exchangeToken = async () => {
                try {
                    // Call our backend to exchange code for token
                    const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/auth/upstox/callback?code=${code}`);
                    if (response.ok) {
                        const data = await response.json();
                        if (data.access_token) {
                            localStorage.setItem('upstox_access_token', data.access_token);
                            // Redirect to home, auth context will detect token
                            navigate('/');
                        }
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
