import type { ReactNode } from "react";

import type { RouteDefinition } from "../routes";

interface AppLayoutProps {
  children: ReactNode;
  routes: RouteDefinition[];
  currentPath: string;
  onNavigate: (path: string) => void;
}

export function AppLayout({ children, routes, currentPath, onNavigate }: AppLayoutProps) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar" aria-label="Main navigation">
        <div className="brand-mark">
          <span className="brand-dot" aria-hidden="true" />
          <span>Smart Exam Proctoring</span>
        </div>
        <nav className="nav-list">
          {routes.map((route) => (
            <button
              className={`nav-link${route.path === currentPath ? " nav-link-active" : ""}`}
              key={route.path}
              onClick={() => onNavigate(route.path)}
              type="button"
            >
              {route.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">Human review remains the final authority.</div>
      </aside>
      <div className="app-content">
        <header className="app-header">
          <div>
            <p className="eyebrow">Local monitoring workspace</p>
            <h1>Exam operations</h1>
          </div>
          <span className="status-pill status-pill-muted">Phase 2 foundation</span>
        </header>
        <main className="app-main">{children}</main>
      </div>
    </div>
  );
}

