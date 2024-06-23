import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useAuth } from "@/lib/auth";
import { Routes } from "@/lib/consts";
import { LogoutService } from "@/api";

export const useSignOut = () => {
  const navigate = useNavigate();
  const { setIsAuth } = useAuth();

  const { mutate, isPending } = useMutation({
    mutationFn: LogoutService.logoutCreate,
    onSuccess() {
      setIsAuth(false);
      navigate({ pathname: Routes.signIn });
    },
  });

  return {
    signOut: mutate,
    isSignOutPending: isPending,
  };
};
