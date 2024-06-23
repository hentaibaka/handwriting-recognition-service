import {
  Control,
  Controller,
  FieldValues,
  RegisterOptions,
} from "react-hook-form";
import { Checkbox } from "../Checkbox";
import { cn } from "@/lib/helpers";
import { ReactNode } from "react";

interface FormFieldProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  control: Control<any>;
  name: string;
  rules: RegisterOptions<FieldValues, string>;
  disabled?: boolean;
  labelSlot?: ReactNode;
}

export const CheckboxFormField = ({
  control,
  name,
  rules,
  disabled,
  labelSlot,
}: FormFieldProps) => {
  return (
    <Controller
      control={control}
      name={name}
      rules={rules}
      render={({ field, fieldState: { error } }) => (
        <div>
          <div className="flex items-center gap-[10px]">
            <Checkbox
              id={name}
              checked={!!field.value}
              className={cn(error && "border-error")}
              disabled={disabled}
              {...field}
              onCheckedChange={(value) =>  field.onChange(value)}
            />
            {labelSlot}
          </div>

          <span className="h-[20px] block text-sm break-words text-error mt-[4px]">
            {error?.message}
          </span>
        </div>
      )}
    />
  );
};
