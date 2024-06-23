import { ReactNode } from "react";
import { Control } from "react-hook-form";
import { CheckboxFormField, FormField } from "@/components";
import { MIN_FIELD_LENGTH, MIN_PASSWORD_LENGTH } from "@/lib/consts";
import { signUpFields } from "../../consts";

interface FormFieldProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  control: Control<any>;
  disabled?: boolean;
  onFieldChange?: (value: string) => void;
}

interface CheckboxFormFieldProps extends FormFieldProps {
  labelSlot?: ReactNode;
}

export const FirstNameFormField = ({
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    control={control}
    name={signUpFields.firstName.name}
    label={signUpFields.firstName.label}
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
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    control={control}
    name={signUpFields.lastName.name}
    label={signUpFields.lastName.label}
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
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    control={control}
    name={signUpFields.middleName.name}
    label={signUpFields.middleName.label}
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
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    control={control}
    name={signUpFields.email.name}
    label={signUpFields.email.label}
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

export const PasswordFormField = ({
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    control={control}
    name={signUpFields.password.name}
    label={signUpFields.password.label}
    rules={{
      required: "Это поле обязательное",
      minLength: {
        value: MIN_PASSWORD_LENGTH,
        message: `Пароль не может быть короче ${MIN_PASSWORD_LENGTH} символов`,
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);

export const RepeatPasswordFormField = ({
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    control={control}
    name={signUpFields.repeatPassword.name}
    label={signUpFields.repeatPassword.label}
    rules={{
      required: "Это поле обязательное",
      validate: (value, formValues) => {
        if (value === formValues[signUpFields.password.name]) return true;
        return "Пароли должны совпадать";
      },
    }}
    disabled={disabled}
    onFieldChange={onFieldChange}
  />
);

export const AgreementTerms = ({
  control,
  disabled,
  labelSlot,
}: CheckboxFormFieldProps) => (
  <CheckboxFormField
    control={control}
    disabled={disabled}
    name={signUpFields.agreementTerms.name}
    labelSlot={labelSlot}
    rules={{
      required: "Это поле обязательное",
    }}
  />
);
