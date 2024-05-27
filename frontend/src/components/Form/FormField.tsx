import {
    Control,
    Controller,
    FieldValues,
    RegisterOptions,
} from "react-hook-form";
import { cn } from "@/lib/helpers";
import { FormFieldContainer } from "./FormFieldContainer";
import { Input } from "../Input";

interface FormFieldProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  control: Control<any>;
  name: string;
  rules: RegisterOptions<FieldValues, string>;
  label: string;
  disabled?: boolean;
  onFieldChange?: (value: string) => void;
}

export const FormField = ({
  control,
  name,
  rules,
  label,
  disabled,
  onFieldChange,
}: FormFieldProps) => (
  <Controller
    control={control}
    name={name}
    rules={rules}
    render={({ field, fieldState: { error } }) => (
      <FormFieldContainer label={label} name={name} errorText={error?.message}>
        <Input
          {...field}
          id={name}
          className={cn(error && "border-error")}
          disabled={disabled}
          onChange={(e) => {
            field.onChange(e);
            onFieldChange?.(e.target.value);
          }}
        />
      </FormFieldContainer>
    )}
  />
);
