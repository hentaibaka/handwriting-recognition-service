import { ProfileService } from "@/api";
import { useAuth } from "@/lib/auth";
import { useQuery } from "@tanstack/react-query";

export type ViewerType = {
  id?: number;
  email?: string;
  firstName?: string;
  lastName?: string;
  middleName?: string;
  status?: string;
};

export const useViewer = () => {
  const { setIsAuth } = useAuth();

  const { data, isLoading, error } = useQuery({
    queryKey: ["viewer"],
    queryFn: ProfileService.profileRetrieve,
    retry: false,
  });

  if (error) {
    setIsAuth(false);
  } else {
    setIsAuth(true);
  }

  return {
    viewerData: {
      id: data?.id,
      email: data?.email,
      firstName: data?.first_name,
      lastName: data?.last_name,
      middleName: data?.middle_name,
      status: data?.status,
    },
    isLoadingViewer: isLoading,
  };
};
