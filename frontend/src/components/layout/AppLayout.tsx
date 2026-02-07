import { Outlet, useLocation } from 'react-router-dom';
import { Navbar } from './Navbar';

export function AppLayout() {
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  return (
    <div className="min-h-screen bg-surface-1">
      <Navbar />
      <main>
        <div className={isHomePage ? 'h-[calc(100vh-3.5rem)] p-3' : 'w-full px-4 lg:px-6 py-4'}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}
