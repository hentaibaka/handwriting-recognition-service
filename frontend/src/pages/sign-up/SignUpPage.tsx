import { cn } from "@/lib/helpers";
import { SignUpForm, SignUpViaVK } from "./components";
import { ComponentWithClassName } from "@/types";
import {isShowVkIDButton} from '@/lib/consts';

export const SignUpPage = () => {
  return (
    <div className="py-[30px] px-[16px] sm:py-[30px] sm:px-[50px] lg:py-[27px] lg:px-[66px]">
      <PageTitle className="mb-[13px]" />
      <SignUpForm
        className="mb-[83px]"
        renderButtonSlot={({ disabled }) => isShowVkIDButton && <SignUpViaVK disabled={disabled} />}
      />
    </div>
  );
};

const PageTitle = ({ className }: ComponentWithClassName) => (
  <div
    className={cn(
      "text-additional-color-2  font-medium",
      "text-[1.1rem] md:text-[1.3rem] lg:text-[1.375rem] lg:leading-[1.75rem]",
      className
    )}
  >
    Регистрация
  </div>
);
