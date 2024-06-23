import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useViewer } from "@/hooks";
import { Spinner } from "@/components";
import { cn } from "@/lib/helpers";
import { useAuth } from "@/lib/auth";
import { Routes } from "@/lib/consts";
import { BreadCrumbs, ProfileForm, SignOutButton } from "./components";
import "./ProfilePage.css";

export const ProfilePage = () => {
  const { viewerData, isLoadingViewer } = useViewer();

  const { isAuth } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isAuth) navigate({ pathname: Routes.signIn });
  }, [isAuth, navigate]);

  return (
    <div className="mb-[40px]">
      <PageTitle />

      <BreadCrumbs
        className="sm:mb-[29px]"
        currentPath={Routes.profile}
        links={breadCrumbsLinks}
      />

      <div className="py-[30px] px-[16px] sm:py-[30px] sm:px-[50px] lg:py-[27px] lg:px-[66px]">
        <CardTitle />

        {!isLoadingViewer ? (
          <ProfileForm
            viewerData={viewerData}
            signOutButton={<SignOutButton />}
          />
        ) : (
          <div className="h-[300px] grid place-content-center">
            <Spinner className="!w-16 !h-16 border-4" />
          </div>
        )}
      </div>
    </div>
  );
};

const PageTitle = () => (
  <h1 className="mb-[16px] font-medium text-[1.2rem] sm:text-[1.875rem] leading-[2.5rem]">
    Личный кабинет
  </h1>
);

const CardTitle = () => (
  <div
    className={cn(
      "mb-[22px] text-additional-color-2 font-medium",
      "text-[1.1rem] md:text-[1.3rem] lg:text-[1.375rem] lg:leading-[1.75rem]"
    )}
  >
    Личные данные
  </div>
);

const breadCrumbsLinks = [
  {
    label: "Главная",
    path: Routes.main,
  },
  {
    label: "Личный кабинет",
    path: Routes.profile,
  },
];
