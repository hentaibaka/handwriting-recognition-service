import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/lib/toast";
import { RequestError } from "@/types";
import { ChangePasswordDto, ProfileService, UpdateUserDto } from "@/api";

export type UpdatedProfileData = UpdateUserDto & ChangePasswordDto;

export const useProfileUpdate = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const {
    isPending: isProfileBaseDataPending,
    mutateAsync: mutateProfileBaseData,
  } = useMutation({
    mutationKey: ["profile-update"],
    mutationFn: ProfileService.profileUpdateUserUpdate,
    onError: (error) => {
      const typedError = error as RequestError;

      if (!typedError.body) console.error(error);

      for (const errorKey in typedError.body) {
        toast({
          title: "Ошибка обновления профиля",
          description: typedError.body[errorKey] || "Неизвестная ошибка",
        });
      }
    },
  });

  const {
    isPending: isProfilePasswordPending,
    mutateAsync: mutateProfilePasswordData,
  } = useMutation({
    mutationFn: ProfileService.profileChangePasswordCreate,
    onError: (error) => {
      const typedError = error as RequestError;

      if (!typedError.body) console.error(error);

      for (const errorKey in typedError.body) {
        toast({
          title: "Ошибка обновления профиля",
          description: typedError.body[errorKey] || "Неизвестная ошибка",
        });
      }
    },
  });

  const updateProfileData = async (updatedProfileData: UpdatedProfileData) => {
    await mutateProfileBaseData(updatedProfileData);

    if (updatedProfileData.new_password && updatedProfileData.old_password)
      await mutateProfilePasswordData(updatedProfileData);

    queryClient.invalidateQueries({ queryKey: ["viewer"] });
  };

  return {
    isProfileDataPending: isProfileBaseDataPending || isProfilePasswordPending,
    updateProfileData,
  };
};
