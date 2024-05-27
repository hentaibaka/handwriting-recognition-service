import { Routes } from "@/lib/consts";
import { BreadCrumbs } from "./components";
import { cn } from "@/lib/helpers";
import { Field } from "@/lib/types";
import { Button, Input, PasswordComplexity } from "@/components";
import "./ProfilePage.css";

const fields: Field[] = [
  {
    name: "status",
    label: "Статус",
  },
  {
    name: "email",
    label: "E-mail",
  },
  {
    name: "lastName",
    label: "Фамилия",
  },
  {
    name: "firstName",
    label: "Имя",
  },
  {
    name: "middleName",
    label: "Отчество",
  },
  {
    name: "password",
    label: "Текущий пароль",
  },
  {
    name: "newPassword",
    label: "Новый пароль",
  },
];

export const ProfilePage = () => {
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

        <div
          className={cn(
            "mb-[38px] profile-page-fields-grid gap-x-[17px] gap-y-[12px] sm:gap-y-[20px] md:gap-y-[30px]",
            // "sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-[repeat(3,_318px)]"
          )}
        >
          {fields.map(({ name, label }) => (
            <div key={name} className={cn('xl:w-[318px]', name)}>
              <label
                htmlFor={name}
                className="mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans"
              >
                {label}
              </label>
              <Input id={name} name={name} />
            </div>
          ))}

          <PasswordComplexity className="mt-[24px] w-full password-complexity md:grid-cols-none xl:grid-cols-[repeat(2,_300px)]" />
        </div>

        <div className="flex flex-wrap flex-col sm:flex-row gap-[12px] sm:gap-[20px]">
          <Button>История распознавания</Button>
          <Button>Редактировать</Button>
          <Button>Выйти</Button>
        </div>
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
