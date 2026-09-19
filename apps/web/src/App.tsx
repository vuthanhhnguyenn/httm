import { useEffect, useMemo, useState } from "react";

import { AppLayout } from "./components/AppLayout";
import { MonitoringPage } from "./pages/MonitoringPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { normalizePath, resolveRoute, type RouteDefinition } from "./routes";

export function App() {
  const [currentPath, setCurrentPath] = useState(() => normalizePath(window.location.pathname));
  const routes = useMemo<RouteDefinition[]>(
    () => [{ path: "/", label: "Monitoring", render: () => <MonitoringPage /> }],
    [],
  );

  useEffect(() => {
    const onPopState = () => setCurrentPath(normalizePath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = (path: string) => {
    window.history.pushState({}, "", path);
    setCurrentPath(normalizePath(path));
  };

  const route = resolveRoute(routes, currentPath);
  return (
    <AppLayout currentPath={currentPath} onNavigate={navigate} routes={routes}>
      {route ? route.render() : <NotFoundPage onBack={() => navigate("/")} />}
    </AppLayout>
  );
}

