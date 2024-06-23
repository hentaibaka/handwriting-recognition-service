import { useMutation } from "@tanstack/react-query";
import { PasswordResetService } from "@/api";

export const useRememberPassword = () => {
  const { mutateAsync, isPending } = useMutation({
    mutationFn: PasswordResetService.passwordResetCreate,
  });

  return {
    rememberPassword: mutateAsync,
    isRememberPasswordPending: isPending,
  };
};
