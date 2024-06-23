import { ReactNode, useState } from "react";
import { useForm } from "react-hook-form";
import { useSignUp } from "@/hooks";
import { Button, PasswordComplexity, PasswordRequirements } from "@/components";
import { cn } from "@/lib/helpers";
import {
  digitRegex,
  MIN_PASSWORD_LENGTH,
  specialCharRegex,
} from "@/lib/consts";
import { RegisterDto } from "@/api";
import {
  AgreementTerms,
  EmailFormField,
  FirstNameFormField,
  LastNameFormField,
  MiddleNameFormField,
  PasswordFormField,
  RepeatPasswordFormField,
} from "../FormFields";
import { signUpFields, SignUpFormFields, termsLink } from "../../consts";

interface SignUpData extends Record<SignUpFormFields, string | boolean> {
  email: string;
  firstName: string;
  lastName: string;
  middleName: string;
  password: string;
  repeatPassword: string;
  agreementTerms: boolean;
}

const defaultValues: SignUpData = {
  firstName: "",
  lastName: "",
  middleName: "",
  email: "",
  password: "",
  repeatPassword: "",
  agreementTerms: false,
};

interface SignUpFormProps {
  renderButtonSlot?: ({ disabled }: { disabled: boolean }) => ReactNode;
  className?: string;
}

interface PasswordComplexityState {
  minPasswordLength: boolean;
  notIncludeNameAndEmail: boolean;
  containsNumberOrSymbol: boolean;
  passwordStrength: "Слабая" | "Нормальная" | "Сильная";
}

export const SignUpForm = ({
  className,
  renderButtonSlot,
}: SignUpFormProps) => {
  const { control, handleSubmit, getValues, trigger } = useForm({
    defaultValues,
    reValidateMode: "onChange",
    mode: "onChange",
  });
  const [passwordComplexity, setPasswordComplexity] =
    useState<PasswordComplexityState>({
      minPasswordLength: false,
      notIncludeNameAndEmail: true,
      containsNumberOrSymbol: false,
      passwordStrength: "Слабая",
    });

  const { isPending, signUp } = useSignUp();

  const getPasswordRequirementsMatch = (password: string) => {
    const isMatchPasswordLength =
      password.length >= MIN_PASSWORD_LENGTH ? true : false;

    const email = getValues(signUpFields.email.name) as string;
    const firstName = getValues(signUpFields.firstName.name) as string;

    const hasEmail = !!email && password.includes(email);
    const hasFirstName = !!firstName && password.includes(firstName);

    const isMatchNotIncludeNameAndEmail = !hasEmail && !hasFirstName;

    const hasDigit = digitRegex.test(password);
    const hasSpecialChar = specialCharRegex.test(password);
    const isMatchContainsNumberOrSymbol = hasDigit || hasSpecialChar;

    return {
      isMatchPasswordLength,
      isMatchNotIncludeNameAndEmail,
      isMatchContainsNumberOrSymbol,
    };
  };

  const triggerRepeatPasswordField = () =>
    trigger(signUpFields.repeatPassword.name);

  const handlePasswordChange = (password: string) => {
    if (password) triggerRepeatPasswordField();

    const {
      isMatchContainsNumberOrSymbol,
      isMatchNotIncludeNameAndEmail,
      isMatchPasswordLength,
    } = getPasswordRequirementsMatch(password);

    const complexityScore = [
      isMatchPasswordLength,
      isMatchNotIncludeNameAndEmail,
      isMatchContainsNumberOrSymbol,
    ].filter(Boolean).length;

    setPasswordComplexity({
      containsNumberOrSymbol: isMatchContainsNumberOrSymbol,
      notIncludeNameAndEmail: isMatchNotIncludeNameAndEmail,
      minPasswordLength: isMatchPasswordLength,
      passwordStrength:
        complexityScore <= 1
          ? "Слабая"
          : complexityScore === 2
          ? "Нормальная"
          : "Сильная",
    });
  };

  const handleFirstNameAndEmailChange = () =>
    handlePasswordChange(getValues(signUpFields.password.name) as string);

  const onSubmit = (data: SignUpData) => {
    const requestData: RegisterDto = {
      email: data.email,
      first_name: data.firstName,
      last_name: data.lastName,
      middle_name: data.middleName,
      password: data.password,
    };

    signUp(requestData);
  };

  return (
    <form className={className} onSubmit={handleSubmit(onSubmit)}>
      <div
        className={cn(
          "mb-[38px] grid gap-x-[17px] gap-y-[12px] sm:gap-y-[20px] md:gap-y-[10px]",
          "sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-[repeat(3,_318px)]"
        )}
      >
        <LastNameFormField disabled={isPending} control={control} />
        <FirstNameFormField
          disabled={isPending}
          control={control}
          onFieldChange={handleFirstNameAndEmailChange}
        />
        <MiddleNameFormField disabled={isPending} control={control} />
        <EmailFormField
          disabled={isPending}
          control={control}
          onFieldChange={handleFirstNameAndEmailChange}
        />
        <PasswordFormField
          disabled={isPending}
          control={control}
          onFieldChange={handlePasswordChange}
        />
        <RepeatPasswordFormField disabled={isPending} control={control} />
      </div>

      <PasswordComplexity className="mb-[38px] grid md:grid-cols-[repeat(2,_300px)] gap-x-[17px] gap-y-[10px]">
        <PasswordRequirements
          isMeetRequirements={passwordComplexity.passwordStrength !== "Слабая"}
          customLabel={
            <div>
              <span>Надежность пароля: </span>
              <span
                className={cn(
                  passwordComplexity.passwordStrength === "Слабая" &&
                    "text-error"
                )}
              >
                {passwordComplexity.passwordStrength}
              </span>
            </div>
          }
        />
        <PasswordRequirements
          isMeetRequirements={passwordComplexity.minPasswordLength}
          label="Минимум 8 символов"
        />
        <PasswordRequirements
          isMeetRequirements={passwordComplexity.notIncludeNameAndEmail}
          label="Не может содержать ваше имя или адрес электронной почты"
        />
        <PasswordRequirements
          isMeetRequirements={passwordComplexity.containsNumberOrSymbol}
          label="Содержит число или символ"
        />
      </PasswordComplexity>

      <div className="mb-[8px]">
        <AgreementTerms
          disabled={isPending}
          control={control}
          labelSlot={
            <div className="text-[0.6875rem] leading-[0.9rem] text-additional-color-7 font-medium">
              <label
                className="block"
                htmlFor={signUpFields.agreementTerms.name}
              >
                Регистрируясь для создания учетной записи, я принимаю
              </label>
              <a
                href={termsLink}
                target="_blank"
                className="font-semibold text-additional-color-4"
              >
                Условия использования и Политику конфиденциальности Компании
              </a>
            </div>
          }
        />
      </div>

      <div className="flex flex-wrap gap-[23px]">
        <Button type="submit" disabled={isPending} className="uppercase">
          Зарегистрироваться
        </Button>

        {renderButtonSlot?.({ disabled: isPending })}
      </div>
    </form>
  );
};
