import { Outlet, useNavigate } from "react-router-dom";
import backgroundImage from "@/assets/images/statisticsBackground.png";
import { cn } from "@/lib/helpers";
import { useAuth } from "@/lib/auth";
import { useEffect } from "react";

export const AuthLayout = () => {
  const { isAuth } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuth) navigate(-1);
  }, [isAuth, navigate]);

  return (
    <>
      <div
        className={cn(
          "mb-[30px] relative overflow-hidden rounded-[20px]",
          "py-[30px] px-[16px] sm:py-[30px] sm:px-[50px] lg:py-[44px] lg:px-[63px]",
          "lg:h-[214px]"
        )}
      >
        <div
          className={cn(
            "text-additional-color-2 sm:w-[322px] font-medium ",
            "text-[1rem] sm:text-[1.2rem] md:text-[1.5rem] lg:text-[1.875rem]",
            "leading-[1.5rem] sm:leading-[2rem] md:leading-[2.2rem] lg:leading-[2.625rem]"
          )}
        >
          Сервис, который{" "}
          <span
            className={cn(
              "font-good-vibes text-accent",
              "text-[1.7rem] sm:text-[2.3rem] md:text-[2.7rem] lg:text-[3.3125rem]"
            )}
          >
            распознает{" "}
          </span>{" "}
          каждую вашу запись
        </div>

        <div className="absolute inset-0 -z-10 bg-gradient-to-r from-white" />
        <div
          className={cn(
            "absolute inset-0 -z-20 opacity-50 mr-[50px] lg:mr-0",
            "bg-[length:90%_280%] bg-no-repeat bg-[left_160%_top_30%] -rotate-[5.5deg]",
            "h-[150px] sm:h-[200px] md:h-[300px]"
          )}
          style={{ backgroundImage: `url(${backgroundImage})` }}
        />
      </div>

      <Outlet />
    </>
  );
};
