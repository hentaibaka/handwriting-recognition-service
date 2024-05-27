import { ReactNode } from "react";

interface FormFieldContainerProps {
  children: ReactNode;
  className?: string;
  name?: string;
  label: string;
  errorText?: string;
}

export const FormFieldContainer = ({
  children,
  className,
  name,
  label,
  errorText,
}: FormFieldContainerProps) => (
  <div className={className}>
    <label
      htmlFor={name}
      className="mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans"
    >
      {label}
    </label>

    {children}

    <span className="h-[20px] block text-sm break-words text-error mt-[4px]">
      {errorText}
    </span>
  </div>
);
