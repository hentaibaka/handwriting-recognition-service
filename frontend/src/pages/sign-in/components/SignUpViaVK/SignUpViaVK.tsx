import { Button, Icon } from "@/components";
import { cn } from "@/lib/helpers";

export const SignUpViaVK = (
  props: React.ButtonHTMLAttributes<HTMLButtonElement>
) => {
  return (
    <Button
      type="button"
      className={cn(
        "bg-additional-color-1 text-accent hover:bg-additional-color-1",
        "disabled:bg-additional-color-1/80 disabled:text-accent/60"
      )}
      {...props}
    >
      <Icon
        iconName="vk-logo"
        className={cn(
          "mr-[10px] h-[30px] w-[30px]",
          props.disabled && "opacity-60"
        )}
      />
      Войти через VK ID
    </Button>
  );
};
