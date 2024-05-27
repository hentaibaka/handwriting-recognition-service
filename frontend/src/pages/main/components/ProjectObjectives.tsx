import { cn } from "@/lib/helpers";

interface ProjectObjectivesProps extends React.HTMLAttributes<HTMLDivElement> {}

export const ProjectObjectives = ({
  className,
  ...props
}: ProjectObjectivesProps) => {
  return (
    <div className={cn(className)} {...props}>
      <div
        className={cn(
          "mb-[30px] w-fit bg-additional-color-1 rounded-[10px] uppercase font-semibold text-accent ",
          "text-[1.5rem] md:text-[1.875rem] md:leading-[28px]",
          "p-[10px] md:p-[18px]"
        )}
      >
        Цель
      </div>

      <div className="flex flex-col sm:flex-row gap-[26px] md:gap-[50px] lg:gap-[76px]">
        <div
          className={cn(
            "md:w-1/2 lg:min-w-[470px] text-additional-color-3 font-medium",
            "text-[1.2rem] md:text-[1.5rem] xl:text-[1.625rem] xl:leading-[1.625rem]"
          )}
        >
          Нашей целью является достижение высокой точности распознавания
          рукописного текста!
        </div>
        <div
          className={cn(
            "md:w-1/2 lg:w-full font-deja-vu-sans text-additional-color-5",
            "md:text-[1.2rem] lg:text-[1.125rem] lg:leading-[1.5rem]"
          )}
        >
          Основная задача нашего проекта – собрать и подготовить обучающий
          датасет, включающий в себя разнообразные изображения с рукописным
          текстом. Затем мы обучаем нашу модель для распознавания рукописного
          текста на изображениях.
        </div>
      </div>
    </div>
  );
};
