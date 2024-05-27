import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Routes } from "@/lib/consts";

export interface SignUpRequestData {
  email: string;
  firstName: string;
  lastName: string;
  middleName: string;
  password: string;
}

export const useSignUp = () => {
  const navigate = useNavigate();

  const { isPending, mutate } = useMutation({
    mutationFn: (data: SignUpRequestData) => {
      return new Promise((res) => setTimeout(() => res(data), 3000));
    },
    onSuccess: () => {
      navigate({ pathname: Routes.recognition });
    },
  });

  return {
    isPending,
    signUp: mutate,
  };
};
