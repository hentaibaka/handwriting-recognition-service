import { useState } from "react";
import { useRememberPassword } from "@/hooks";
import {
  Button,
  Input,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  Checkbox,
} from "@/components";

const rememberPasswordInputId = "rememberPassword";

export const RememberPasswordModal = () => {
  const [email, setEmail] = useState("");
  const [isPasswordSended, setIsPasswordSended] = useState(false);
  const { isRememberPasswordPending, rememberPassword } = useRememberPassword();

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEmail(event.target.value);
  };

  const handleButtonClick = () => {
    rememberPassword({ email }).then(() => setIsPasswordSended(true));
  };

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
              htmlFor={rememberPasswordInputId}
              className="mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans"
            >
              E-mail
            </label>
            <Input
              id={rememberPasswordInputId}
              name="email"
              onChange={handleInputChange}
            />
          </div>

          <Button
            className="uppercase"
            disabled={isRememberPasswordPending}
            onClick={handleButtonClick}
          >
            Отправить
          </Button>
        </div>

        {isPasswordSended && (
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
        )}
      </DialogContent>
    </Dialog>
  );
};
