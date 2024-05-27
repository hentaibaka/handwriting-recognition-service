import { cn } from "@/lib/helpers";
import { ComponentWithClassName } from "@/types";

export const PageTitle = ({ className }: ComponentWithClassName) => (
  <h2
    className={cn(
      "text-[1.1rem] md:text-[1.5rem] md:leading-[2rem] lg:text-[30px] lg:leading-[40px] font-medium uppercase max-w-[700px]",
      className
    )}
  >
    Сервис преобразования рукописного текста в{" "}
    <span className="text-accent">цифровой формат</span>
  </h2>
);
