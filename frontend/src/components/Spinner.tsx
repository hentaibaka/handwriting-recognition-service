import clsx from "clsx";

interface SpinnerProps {
  className?: string;
}

export const Spinner = ({ className }: SpinnerProps) => {
  return (
    <div
      className={clsx(
        "w-5 h-5 inline-block border-2 !border-l-accent !border-t-accent border-accent/50 animate-spin rounded-full",
        className
      )}
      role="status"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
};
