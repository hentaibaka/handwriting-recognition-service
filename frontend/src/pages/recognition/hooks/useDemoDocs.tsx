import { DemodocsService } from "@/api";
import { useQuery } from "@tanstack/react-query";

export const useDemoDocs = () => {
  const { data, isLoading } = useQuery({
    queryKey: ["demodocs"],
    queryFn: DemodocsService.demodocsList,
  });

  return {
    demoDocs: data,
    isDemoDocsLoading: isLoading,
  };
};
