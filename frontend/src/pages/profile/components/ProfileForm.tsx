import { ReactNode, useState } from "react";
import { useForm } from "react-hook-form";
import { UpdatedProfileData, useProfileUpdate, ViewerType } from "@/hooks";
import {
  Button,
  Input,
  PasswordComplexity,
  PasswordRequirements,
} from "@/components";
import {
  digitRegex,
  MIN_PASSWORD_LENGTH,
  specialCharRegex,
} from "@/lib/consts";
import { cn } from "@/lib/helpers";
import {
  EmailFormField,
  LastNameFormField,
  FirstNameFormField,
  MiddleNameFormField,
  CurrentPasswordFormField,
  NewPasswordFormField,
} from "./FormFields";
import { ProfileFormFields, profileFormFields } from "../consts";

interface ProfileFormData extends Record<ProfileFormFields, string | boolean> {
  status: string;
  email: string;
  firstName: string;
  lastName: string;
  middleName: string;
  currentPassword: string;
  newPassword: string;
}

interface ProfileFormProps {
  viewerData: ViewerType;
  signOutButton: ReactNode;
}

interface PasswordComplexityState {
  minPasswordLength: boolean;
  notIncludeNameAndEmail: boolean;
  containsNumberOrSymbol: boolean;
  passwordStrength: "Слабая" | "Нормальная" | "Сильная";
}

export const ProfileForm = ({
  viewerData,
  signOutButton,
}: ProfileFormProps) => {
  const defaultValues: ProfileFormData = {
    email: viewerData.email || "",
    firstName: viewerData.firstName || "",
    lastName: viewerData.lastName || "",
    middleName: viewerData.middleName || "",
    currentPassword: "",
    newPassword: "",
    status: "",
  };

  const { control, handleSubmit, getValues } = useForm({
    defaultValues,
    reValidateMode: "onChange",
    mode: "onChange",
  });

  const { isProfileDataPending, updateProfileData } = useProfileUpdate();

  const [passwordComplexity, setPasswordComplexity] =
    useState<PasswordComplexityState>({
      minPasswordLength: false,
      notIncludeNameAndEmail: true,
      containsNumberOrSymbol: false,
      passwordStrength: "Слабая",
    });
  const [isShowPasswordComplexity, setIsShowPasswordComplexity] =
    useState(false);

  const getPasswordRequirementsMatch = (password: string) => {
    const isMatchPasswordLength =
      password.length >= MIN_PASSWORD_LENGTH;

    const email = getValues(profileFormFields.email.name) as string;
    const firstName = getValues(profileFormFields.firstName.name) as string;

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

  const handlePasswordChange = (password: string) => {
    setIsShowPasswordComplexity(true);

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

  const onSubmit = (data: ProfileFormData) => {
    const requestData: UpdatedProfileData = {
      email: data.email,
      first_name: data.firstName,
      last_name: data.lastName,
      middle_name: data.middleName,
      new_password: data.newPassword,
      old_password: data.currentPassword,
    };

    updateProfileData(requestData);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="mb-[38px] profile-page-fields-grid gap-x-[17px] gap-y-[12px] sm:gap-y-[20px] md:gap-y-[30px]">
        <div className="xl:w-[318px] status">
          <label className="mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans">
            Статус
          </label>
          <Input className="!cursor-default border-accent" disabled value={viewerData.status} />
        </div>

        <EmailFormField
          className="xl:w-[318px] email"
          control={control}
          disabled={isProfileDataPending}
        />
        <LastNameFormField
          className="xl:w-[318px] lastName"
          control={control}
          disabled={isProfileDataPending}
        />
        <FirstNameFormField
          className="xl:w-[318px] firstName"
          control={control}
          disabled={isProfileDataPending}
        />
        <MiddleNameFormField
          className="xl:w-[318px] middleName"
          control={control}
          disabled={isProfileDataPending}
        />
        <CurrentPasswordFormField
          className="xl:w-[318px] password"
          control={control}
          disabled={isProfileDataPending}
          onFieldChange={handlePasswordChange}
        />
        <NewPasswordFormField
          className="xl:w-[318px] newPassword"
          control={control}
          disabled={isProfileDataPending}
          onFieldChange={handlePasswordChange}
        />

        <PasswordComplexity
          className={cn(
            "password-complexity",
            !isShowPasswordComplexity && "invisible"
          )}
        >
          <PasswordRequirements
            isMeetRequirements={
              passwordComplexity.passwordStrength !== "Слабая"
            }
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
      </div>

      <div className="flex flex-wrap flex-col sm:flex-row gap-[12px] sm:gap-[20px]">
        <Button type="button">История распознавания</Button>
        <Button type="submit">Редактировать</Button>
        {signOutButton}
      </div>
    </form>
  );
};
