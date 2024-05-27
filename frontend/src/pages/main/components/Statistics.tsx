import { Card } from "@/components";
import { cn } from "@/lib/helpers";
import backgroundImage from "@/assets/images/statisticsBackground.png";
import { useMetrics } from "../hooks";

interface SectionsProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Statistics = ({ className, ...props }: SectionsProps) => {
  const { metrics, isMetricsLoading } = useMetrics();

  return (
    <div
      className={cn(
        "py-[26px] px-[6px] md:py-[45px] md:px-[30px] relative overflow-clip",
        className
      )}
      {...props}
    >
      <h3 className="mb-[36px] uppercase font-semibold text-additional-color-2 text-[1.2rem] sm:text-[1.5rem] md:text-[1.875rem]">
        Статистика:
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-[repeat(5,_200px)] gap-[41px]">
        {isMetricsLoading && <MetricsSkeleton />}

        {!isMetricsLoading &&
          metrics?.map(({ name, value }) => (
            <div className="font-medium w-fit" key={name}>
              <div className="text-[1.8rem] sm:text-[2rem] md:text-[55px] md:leading-[55px] font-bold text-accent">
                {value}
              </div>
              <div className="text-[0.8rem] md:text-[14px] md:leading-[18px] text-additional-color-4">
                {name}
              </div>
            </div>
          ))}
      </div>

      <div className="absolute inset-0 -z-10 bg-gradient-to-r from-white via-white" />

      <div
        className="absolute inset-0 -z-20 rounded-[10px] h-[350px] xl:h-[400px] bg-no-repeat opacity-30 bg-center md:bg-[length:115%_248%] md:bg-[right_250%_top_25%] -rotate-[8deg]"
        style={{ backgroundImage: `url(${backgroundImage})` }}
      />
    </div>
  );
};

const MetricsSkeleton = () =>
  [1, 2, 3, 4, 5].map((num) => (
    <Card
      key={num}
      className="h-[91px] w-[190px] bg-neutral-100 animate-pulse rounded-[10px]"
    />
  ));
