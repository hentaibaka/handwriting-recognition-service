import { cn } from "@/lib/helpers";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Card = ({ children, className, ...props }: CardProps) => {
  return (
    <div
      className={cn(
        "rounded-[20px] px-[45px] py-[43px] bg-white shadow-card",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
