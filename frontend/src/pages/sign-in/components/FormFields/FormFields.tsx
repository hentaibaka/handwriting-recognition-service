import { Control } from "react-hook-form";
import { FormField } from "@/components";
import { MIN_FIELD_LENGTH, MIN_PASSWORD_LENGTH } from "@/lib/consts";
import { signInFields } from "../../consts";

interface FormFieldProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  control: Control<any>;
  disabled?: boolean;
  onFieldChange?: (value: string) => void;
}

export const EmailFormField = ({
  control,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <FormField
    control={control}
    name={signInFields.email.name}
    label={signInFields.email.label}
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
    name={signInFields.password.name}
    label={signInFields.password.label}
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
