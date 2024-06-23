import { cn } from "@/lib/helpers";

export const Container = ({
  children,
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn("mx-auto w-full max-w-[1320px] px-2 xl:px-0", className)}
    {...props}
  >
    {children}
  </div>
);
