import { cn } from "@/lib/helpers";
import { Icon } from "./Icon";
import { ReactNode } from "react";

interface PasswordComplexityProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export const PasswordComplexity = ({
  className,
  ...props
}: PasswordComplexityProps) => {
  return (
    <div
      className={cn(
        "grid md:grid-cols-[repeat(2,_300px)] gap-x-[17px] gap-y-[10px]",
        className
      )}
      {...props}
    >
      {/* <div className="flex items-center gap-[9px] text-[0.75rem] leading-[0.75rem] font-deja-vu-sans text-additional-color-4">
        <Icon
          iconName="check"
          className="min-w-4 max-w-4 min-h-4 max-h-4 text-error"
        />
        <div>
          <span>Надежность пароля: </span>
          <span className="text-error">Слабая</span>
        </div>
      </div>

      <div className="flex items-center gap-[9px] text-[0.75rem] leading-[0.75rem] font-deja-vu-sans text-additional-color-4">
        <Icon
          iconName="check"
          className="min-w-4 max-w-4 min-h-4 max-h-4 text-accent/30"
        />
        <span>Минимум 8 символов</span>
      </div>

      <div className="flex items-center gap-[9px] text-[0.75rem] leading-[0.75rem] font-deja-vu-sans text-additional-color-4">
        <Icon
          iconName="check"
          className="min-w-4 max-w-4 min-h-4 max-h-4 text-accent/30"
        />
        <span>Не может содержать ваше имя или адрес электронной почты</span>
      </div>

      <div className="flex items-center gap-[9px] text-[0.75rem] leading-[0.75rem] font-deja-vu-sans text-additional-color-4">
        <Icon
          iconName="check"
          className="min-w-4 max-w-4 min-h-4 max-h-4 text-accent/30"
        />
        <span>Содержит число или символ</span>
      </div> */}
    </div>
  );
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
