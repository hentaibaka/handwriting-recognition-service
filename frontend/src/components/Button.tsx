import React from "react";
import clsx from "clsx";
import { VariantProps, cva } from "class-variance-authority";
import { cn } from "@/lib/helpers";

const buttonVariants = cva(
  clsx(
    "inline-flex w-fit text-nowrap items-center justify-center rounded-[10px] duration-300 outline-none",
    "font-gilroy text-base font-normal tracking-[0.6px]",
    "text-[0.8rem] md:text-[1rem] lg:text-[1.25rem]",
    "focus-visible:ring ring-accent ring-offset-2",
    "active:scale-95 disabled:scale-100 disabled:cursor-not-allowed"
  ),
  {
    variants: {
      variant: {
        default:
          "bg-accent text-white hover:bg-accent disabled:bg-accent/50 disabled:text-white/70",
        outline:
          "border border-accent text-accent bg-transparent disabled:text-white/70",
        ghost: "hover:text-additional-color-2/70",
        link: "text-accent underline-offset-4 underline",
      },
      size: {
        default:
          "h-[42px] p-[10px] md:h-[50px] md:p-[12px] lg:h-[57px] lg:p-[15px] ",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  )
);

Button.displayName = "Button";
