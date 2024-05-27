import { Button, Card } from "@/components";
import { cn } from "@/lib/helpers";
import systemSectionImg from "@/assets/images/systemSectionImg.png";
import systemSectionEllipse1 from "@/assets/images/systemSectionEllipse1.png";
import systemSectionEllipse2 from "@/assets/images/systemSectionEllipse2.png";
import { Link } from "react-router-dom";
import { Routes } from "@/lib/consts";

interface UseSystemProps extends React.HTMLAttributes<HTMLDivElement> {}

export const UseSystem = ({ className }: UseSystemProps) => {
  return (
    <Card
      className={cn(
        "bg-additional-color-1 relative overflow-hidden",
        "px-[10px] py-[12px] sm:p-[18px] md:px-[45px] md:py-[43px]",
        className
      )}
    >
      <div className="max-w-[640px] z-50 relative">
        <h4 className="mb-[11px] text-additional-color-3 font-medium md:text-[1.25rem] md:leading-[1.25rem]">
          Каждое слово – ключ к новым возможностям!
        </h4>
        <div className="mb-[20px] text-additional-color-5 font-deja-vu-sans md:text-[1.125rem] md:leading-[1.5rem]">
          Не теряйте времени, присоединяйтесь к нашей платформе прямо сейчас!
          Вместе мы расширим границы вашего понимания текста
        </div>

        <Link to={{ pathname: Routes.recognition }}>
          <Button className="uppercase">Использовать систему</Button>
        </Link>
      </div>

      <div className="hidden lg:block absolute h-full w-[450px] top-0 right-[10%]">
        <div
          className="w-full h-full bg-no-repeat absolute top-[15%] z-30 pointer-events-none"
          style={{ backgroundImage: `url(${systemSectionImg})` }}
        />
        <div
          className="w-full h-full bg-no-repeat absolute bg-top -left-16 z-20 pointer-events-none"
          style={{ backgroundImage: `url(${systemSectionEllipse1})` }}
        />
        <div
          className="w-full h-full bg-no-repeat absolute bg-top -left-10 z-10 pointer-events-none"
          style={{ backgroundImage: `url(${systemSectionEllipse2})` }}
        />
      </div>
    </Card>
  );
};
