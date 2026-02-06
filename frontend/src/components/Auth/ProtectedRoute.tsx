import { ReactNode } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import Login from './Login';

interface ProtectedRouteProps {
    children: ReactNode;
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
    const { isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
        return (
            <div style={{
                width: '100%',
                height: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#0f172a',
                color: 'white'
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{
                        width: 50,
                        height: 50,
                        border: '3px solid rgba(255, 255, 255, 0.1)',
                        borderTop: '3px solid #3b82f6',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite',
                        margin: '0 auto 20px'
                    }} />
                    <h2>Loading...</h2>
                </div>
            </div>
        );
    }

    if (!isAuthenticated) {
        return <Login />;
    }

    return <>{children}</>;
};

export default ProtectedRoute;
