import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

interface AuthContextType {
    isAuthenticated: boolean;
    isLoading: boolean;
    login: () => void;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // Check if user has valid token on mount and listen for storage changes
    useEffect(() => {
        const checkAuth = () => {
            const token = localStorage.getItem('upstox_access_token');
            setIsAuthenticated(!!token);
            setIsLoading(false);
        };

        // Check on mount
        checkAuth();

        // Listen for storage changes (from other tabs or when callback sets token)
        const handleStorageChange = (e: StorageEvent) => {
            if (e.key === 'upstox_access_token') {
                setIsAuthenticated(!!e.newValue);
            }
        };

        // Listen for window focus to re-check auth (handles redirect from OAuth)
        const handleFocus = () => {
            checkAuth();
        };

        window.addEventListener('storage', handleStorageChange);
        window.addEventListener('focus', handleFocus);

        return () => {
            window.removeEventListener('storage', handleStorageChange);
            window.removeEventListener('focus', handleFocus);
        };
    }, []);

    const login = () => {
        // Redirect to backend login endpoint
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        window.location.href = `${apiUrl}/auth/upstox/login`;
    };

    const logout = () => {
        localStorage.removeItem('upstox_access_token');
        setIsAuthenticated(false);
        window.location.href = '/';
    };

    return (
        <AuthContext.Provider value={{ isAuthenticated, isLoading, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
};
