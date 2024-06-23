import { Button } from "@/components";

interface CopyTextButton {
  text: string | null;
}

export const CopyTextButton = ({ text }: CopyTextButton) => {

  const handleCopyText = async () => {
    if (!text) return;

    
  };

  return (
    <Button
      className="font-deja-vu-sans"
      disabled={!text}
      onClick={handleCopyText}
    >
      Скопировать
    </Button>
  );
};
