import { RouterProvider } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster, useToast } from "@/lib/toast";
import { AuthProvider } from "@/lib/auth";
import { RequestError } from "@/types";
import { router } from "./router";

export const App = () => {
  const { toast } = useToast();
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
      },
      mutations: {
        onError(error) {
          const typedError = error as RequestError;

          if (!typedError.body) console.error(error);

          for (const errorKey in typedError.body) {
            toast({
              title: "Ошибка",
              description: typedError.body[errorKey] || "Неизвестная ошибка",
            });
          }
        },
      },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <RouterProvider router={router} />
      </AuthProvider>
      <Toaster />
    </QueryClientProvider>
  );
};
