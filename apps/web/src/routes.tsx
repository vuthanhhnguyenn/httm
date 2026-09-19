import type { ReactNode } from "react";

export interface RouteDefinition {
  path: string;
  label: string;
  render: () => ReactNode;
}

export const normalizePath = (value: string): string => {
  const path = value.replace(/\/+$/, "");
  return path || "/";
};

export const resolveRoute = (routes: RouteDefinition[], path: string): RouteDefinition | undefined => {
  const normalized = normalizePath(path);
  return routes.find((route) => normalizePath(route.path) === normalized);
};

