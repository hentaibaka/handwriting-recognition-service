import { MetricsService } from "@/api";
import { useQuery } from "@tanstack/react-query";

export const useMetrics = () => {
  const { data, isLoading } = useQuery({
    queryKey: ["metrics"],
    queryFn: MetricsService.metricsList,
  });

  return {
    metrics: data,
    isMetricsLoading: isLoading,
  };
};
