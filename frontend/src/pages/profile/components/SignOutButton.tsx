import { useSignOut } from "@/hooks";
import { Button } from "@/components";

export const SignOutButton = () => {
  const { isSignOutPending, signOut } = useSignOut();

  return (
    <Button type="button" disabled={isSignOutPending} onClick={() => signOut()}>
      Выйти
    </Button>
  );
};
