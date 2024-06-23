import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { LoginDto, LoginService } from "@/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";
import { RequestError } from "@/types";

export const useSignIn = () => {
  const navigate = useNavigate();
  const { setIsAuth } = useAuth();
  const { toast } = useToast();

  const { isPending, mutate } = useMutation({
    mutationFn: (data: LoginDto) => LoginService.loginCreate(data),
    onSuccess: (response) => {
      setIsAuth(true);
      navigate({ pathname: response.redirect_url });
      window.location.reload();
    },
    onError: (error) => {
      const typedError = error as RequestError;

      if (!typedError.body) console.error(error);

      for (const errorKey in typedError.body) {
        toast({
          title: "Ошибка авторизации",
          description: typedError.body[errorKey] || "Неизвестная ошибка",
        });
      }
    },
  });

  return {
    isPending,
    signIn: mutate,
  };
};
