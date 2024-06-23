import { lazy } from "react";
import { createBrowserRouter, Navigate, RouteObject } from "react-router-dom";
import { AuthLayout, MainLayout } from "@/layouts";
import { Routes } from "@/lib/consts";
const MainPage = lazy(() => import("@/pages/main"));
const ProfilePage = lazy(() => import("@/pages/profile"));
const RecognitionPage = lazy(() => import("@/pages/recognition"));
const SignInPage = lazy(() => import("@/pages/sign-in"));
const SignUpPage = lazy(() => import("@/pages/sign-up"));

export const routesConfig: RouteObject[] = [
  {
    element: <MainLayout />,
    children: [
      {
        index: true,
        path: Routes.main,
        element: <MainPage />,
      },
      {
        path: Routes.profile,
        element: <ProfilePage />,
      },
      {
        path: Routes.recognition,
        element: <RecognitionPage />,
      },
      {
        element: <AuthLayout />,
        children: [
          {
            path: Routes.signIn,
            element: <SignInPage />,
          },
          {
            path: Routes.signUp,
            element: <SignUpPage />,
          },
        ],
      },
    ],
  },
  {
    path: "/admin",
  },
  { path: "*", element: <Navigate to={Routes.main} replace /> },
];

export const router = createBrowserRouter(routesConfig);
