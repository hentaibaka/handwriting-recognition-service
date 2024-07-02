import { SVGAttributes } from "react";
import { cn } from "@/lib/helpers";

export type IconNames =
  | "logo"
  | "logo-footer"
  | "sibFU-logo"
  | "IKIT-logo"
  | "AI-center-logo"
  | "vk-logo"
  | "email"
  | "check"
  | "close"
  | "caretLeft"
  | "caretRight";

interface IconProps extends SVGAttributes<SVGSVGElement> {
  iconName?: IconNames;
}

export const Icon = ({ iconName, className, ...props }: IconProps) => {
  const fileName = "iconsSprite.svg";

  return (
    <svg
      focusable="false"
      aria-hidden
      className={cn(
        "select-none fill-current inline-block text-inherit text-center w-6 h-6 outline-none",
        className
      )}
      {...props}
    >
      <use xlinkHref={`${fileName}#${iconName}`} />
    </svg>
  );
};
