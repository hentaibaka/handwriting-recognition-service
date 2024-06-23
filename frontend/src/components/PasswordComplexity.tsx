import { cn } from "@/lib/helpers";
import { Icon } from "./Icon";
import { ReactNode } from "react";

interface PasswordComplexityProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export const PasswordComplexity = ({
  className,
  ...props
}: PasswordComplexityProps) => {
  return <div className={cn(className)} {...props} />;
};

interface PasswordRequirementsProps {
  label?: string;
  customLabel?: ReactNode;
  isMeetRequirements: boolean;
}

export const PasswordRequirements = ({
  label,
  customLabel,
  isMeetRequirements,
}: PasswordRequirementsProps) => (
  <div className="flex items-center gap-[9px] text-[0.75rem] leading-[0.75rem] font-deja-vu-sans">
    <Icon
      iconName="check"
      className={cn(
        "min-w-4 max-w-4 min-h-4 max-h-4",
        isMeetRequirements ? "text-accent/30" : "text-error"
      )}
    />
    {customLabel ? (
      <>{customLabel}</>
    ) : (
      <span
        className={cn(
          isMeetRequirements ? "text-additional-color-4" : "text-error"
        )}
      >
        {label}
      </span>
    )}
  </div>
);
