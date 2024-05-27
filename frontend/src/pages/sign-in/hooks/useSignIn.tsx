import { Routes } from "@/lib/consts";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

export interface SignInRequestData {
  email: string;
  password: string;
}

export const useSignIn = () => {
  const navigate = useNavigate();

  const { isPending, mutate } = useMutation({
    mutationFn: (data: SignInRequestData) => {
      return new Promise((res) => setTimeout(() => res(data), 3000));
    },
    onSuccess: () => {
      navigate({ pathname: Routes.recognition });
    },
  });

  return {
    isPending,
    signIn: mutate,
  };
};
