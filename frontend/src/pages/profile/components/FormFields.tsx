import { Control } from "react-hook-form";
import { FormField } from "@/components";
import { MIN_FIELD_LENGTH, MIN_PASSWORD_LENGTH } from "@/lib/consts";
import { profileFormFields } from "../consts";

interface FormFieldProps {
  className?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  control: Control<any>;
  disabled?: boolean;
  onFieldChange?: (value: string) => void;
}

export const FirstNameFormField = ({
  className,
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    className={className}
    control={control}
    name={profileFormFields.firstName.name}
    label={profileFormFields.firstName.label}
    rules={{
      required: "Это поле обязательное",
      minLength: {
        value: MIN_FIELD_LENGTH,
        message: `Поле не может быть короче ${MIN_FIELD_LENGTH} символов`,
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);

export const LastNameFormField = ({
  className,
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    className={className}
    control={control}
    name={profileFormFields.lastName.name}
    label={profileFormFields.lastName.label}
    rules={{
      required: "Это поле обязательное",
      minLength: {
        value: MIN_FIELD_LENGTH,
        message: `Поле не может быть короче ${MIN_FIELD_LENGTH} символов`,
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);

export const MiddleNameFormField = ({
  className,
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    className={className}
    control={control}
    name={profileFormFields.middleName.name}
    label={profileFormFields.middleName.label}
    rules={{
      required: "Это поле обязательное",
      minLength: {
        value: MIN_FIELD_LENGTH,
        message: `Поле не может быть короче ${MIN_FIELD_LENGTH} символов`,
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);

export const EmailFormField = ({
  className,
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    className={className}
    control={control}
    name={profileFormFields.email.name}
    label={profileFormFields.email.label}
    rules={{
      required: "Это поле обязательное",
      minLength: {
        value: MIN_FIELD_LENGTH,
        message: `Поле не может быть короче ${MIN_FIELD_LENGTH} символов`,
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);

export const CurrentPasswordFormField = ({
  className,
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    className={className}
    control={control}
    name={profileFormFields.currentPassword.name}
    label={profileFormFields.currentPassword.label}
    rules={{
      minLength: {
        value: MIN_PASSWORD_LENGTH,
        message: `Пароль не может быть короче ${MIN_PASSWORD_LENGTH} символов`,
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);

export const NewPasswordFormField = ({
  className,
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    className={className}
    control={control}
    name={profileFormFields.newPassword.name}
    label={profileFormFields.newPassword.label}
    rules={{
      minLength: {
        value: MIN_PASSWORD_LENGTH,
        message: `Пароль не может быть короче ${MIN_PASSWORD_LENGTH} символов`,
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);
