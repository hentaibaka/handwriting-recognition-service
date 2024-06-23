import { cn } from "@/lib/helpers";
import { RememberPasswordModal, SignInForm, SignUpViaVK } from "./components";
import { isShowVkIDButton } from "@/lib/consts";

export const SignInPage = () => {
  return (
    <div className="py-[30px] px-[16px] sm:py-[30px] sm:px-[50px] lg:py-[27px] lg:px-[66px]">
      <PageTitle />

      <SignInForm
        renderButtonSlot={({ disabled }) =>
          isShowVkIDButton && <SignUpViaVK disabled={disabled} />
        }
      />

      <RememberPasswordModal />
    </div>
  );
};

const PageTitle = () => (
  <div
    className={cn(
      "mb-[13px] text-additional-color-2 font-medium",
      "text-[1.1rem] md:text-[1.3rem] lg:text-[1.375rem] lg:leading-[1.75rem]"
    )}
  >
    Авторизация
  </div>
);
