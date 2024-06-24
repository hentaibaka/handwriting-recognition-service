import { useState } from "react";
import { Button } from "@/components";

interface CopyTextButtonProps {
  text: string | null;
}

export const CopyTextButton = ({ text }: CopyTextButtonProps) => {
  const [buttonText, setButtonText] = useState("Скопировать");
  const [isCopied, setIsCopied] = useState(false);

  const handleCopyText = async () => {
    if (!text) return;

    const unsecuredCopyToClipboard = (text: string) => {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
      } catch (err) {
        console.error('Unable to copy to clipboard', err);
      }
      document.body.removeChild(textArea);
    }

    text = text.replace(/<br>/g, '\n');

    try {
      if (window.isSecureContext && navigator.clipboard) {
        navigator.clipboard.writeText(text);
      } else {
        unsecuredCopyToClipboard(text);
      }
      isCopied;
      setButtonText("Скопировано");
      setIsCopied(true);
      setTimeout(() => {
        setButtonText("Скопировать");
        setIsCopied(false);
      }, 2000); // Возвращаем исходный текст через 2 секунды
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Button
      className="font-deja-vu-sans"
      disabled={!text}
      onClick={handleCopyText}
    >
      {buttonText}
    </Button>
  );
};
