import {
  Button,
  Checkbox,
  Input,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components";
import { cn } from "@/lib/helpers";
import { SignInForm, SignUpViaVK } from "./components";

const rememberPassword = "rememberPassword";

export const SignInPage = () => {
  return (
    <div className="py-[30px] px-[16px] sm:py-[30px] sm:px-[50px] lg:py-[27px] lg:px-[66px]">
      <PageTitle />

      <SignInForm
        renderButtonSlot={({ disabled }) => <SignUpViaVK disabled={disabled} />}
      />

      <RememberPassword />
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

const RememberPassword = () => {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="link" className="!p-0 !py-2 !h-fit !text-[1rem]">
          Забыли пароль?
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-[708px]">
        <DialogHeader>
          <DialogTitle>Забыли пароль?</DialogTitle>

          <DialogDescription>
            Мы отправим вам новый пароль на вашу почту
          </DialogDescription>
        </DialogHeader>

        <div className="mb-[20px] flex flex-wrap sm:flex-nowrap items-end gap-[13px]">
          <div className="w-full">
            <label
              htmlFor={rememberPassword}
              className="mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans"
            >
              E-mail
            </label>
            <Input id={rememberPassword} name={rememberPassword} />
          </div>

          <Button className="uppercase">Отправить</Button>
        </div>

        <div className="bg-success p-[14px] rounded-[10px] font-deja-vu-sans flex items-center gap-[8px]">
          <Checkbox
            disabled
            checked
            className="rounded-[6px] !bg-green border-0 cursor-default"
          />
          <div className="text-[0.8rem] sm:text-[1rem]">
            Новый пароль был отправлен
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
