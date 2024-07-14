import { Button, Icon } from "@/components";
import { cn } from "@/lib/helpers";

const vkAuthUrl = `https://oauth.vk.com/authorize?client_id=51986588&redirect_uri=https://manuscript.sfu-kras.ru/auth/complete/vk-oauth2/&response_type=code&scope=email`;

export const SignUpViaVK = (
  props: React.ButtonHTMLAttributes<HTMLButtonElement>
) => {
  const handleVKLogin = () => {
    window.location.href = vkAuthUrl;
  };

  return (
    <Button
      type="button"
      className={cn(
        "bg-additional-color-1 text-accent hover:bg-additional-color-1",
        "disabled:bg-additional-color-1/80 disabled:text-accent/60"
      )}
      onClick={handleVKLogin}
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
