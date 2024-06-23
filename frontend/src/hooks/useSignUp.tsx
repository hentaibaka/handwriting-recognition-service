import { useMutation } from "@tanstack/react-query";
import { useToast } from "@/lib/toast";
import { RequestError } from "@/types";
import { RegisterDto, RegisterService } from "@/api";
import { useSignIn } from "./useSignIn";

export const useSignUp = () => {
  const { isPending: isPendingSignUp, signIn } = useSignIn();
  const { toast } = useToast();

  const { isPending: isPendingSignIn, mutate } = useMutation({
    mutationFn: (data: RegisterDto) => RegisterService.registerCreate(data),
    onSuccess: (_, signUpData) => {
      signIn({
        email: signUpData.email,
        password: signUpData.password,
      });
    },
    onError: (error) => {
      const typedError = error as RequestError;

      if (!typedError.body) console.error(error);

      for (const errorKey in typedError.body) {
        toast({
          title: "Ошибка регистрации",
          description: typedError.body[errorKey] || "Неизвестная ошибка",
        });
      }
    },
  });

  return {
    isPending: isPendingSignIn || isPendingSignUp,
    signUp: mutate,
  };
};
