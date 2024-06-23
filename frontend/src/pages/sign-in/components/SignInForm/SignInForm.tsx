import { ReactNode } from "react";
import { useForm } from "react-hook-form";
import { Button } from "@/components";
import { useSignIn } from "@/hooks";
import { LoginDto } from "@/api";
import { cn } from "@/lib/helpers";
import { EmailFormField, PasswordFormField } from "../FormFields";
import { SignInFormFields } from "../../consts";

interface SignInData extends Record<SignInFormFields, string | boolean> {
  email: string;
  password: string;
}

const defaultValues: SignInData = {
  email: "",
  password: "",
};

interface SignInFormProps {
  renderButtonSlot?: ({ disabled }: { disabled: boolean }) => ReactNode;
  className?: string;
}

export const SignInForm = ({
  className,
  renderButtonSlot,
}: SignInFormProps) => {
  const { control, handleSubmit } = useForm({
    defaultValues,
  });

  const { isPending, signIn } = useSignIn();

  const onSubmit = (data: SignInData) => {
    const requestData: LoginDto = {
      email: data.email,
      password: data.password,
    };

    signIn(requestData);
  };

  return (
    <form className={className} onSubmit={handleSubmit(onSubmit)}>
      <div
        className={cn(
          "mb-[26px] grid gap-x-[17px] gap-y-[12px] sm:gap-y-[20px] md:gap-y-[30px]",
          "sm:grid-cols-2 xl:grid-cols-[repeat(2,_403px)]"
        )}
      >
        <EmailFormField disabled={isPending} control={control} />
        <PasswordFormField disabled={isPending} control={control} />
      </div>

      <div className="mb-[26px] flex flex-wrap gap-[23px]">
        <Button className="uppercase" disabled={isPending}>
          Авторизоваться
        </Button>
        {renderButtonSlot?.({ disabled: isPending })}
      </div>
    </form>
  );
};
