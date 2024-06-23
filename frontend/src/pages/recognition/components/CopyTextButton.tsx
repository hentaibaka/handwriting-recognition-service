import React, { useState } from "react";
import { Button } from "@/components";

interface CopyTextButtonProps {
  text: string | null;
}

export const CopyTextButton = ({ text }: CopyTextButtonProps) => {
  const [buttonText, setButtonText] = useState("Скопировать");
  const [isCopied, setIsCopied] = useState(false);

  const handleCopyText = async () => {
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      setButtonText("Скопировано");
      setIsCopied(true);
      setTimeout(() => {
        setButtonText("Скопировать");
        setIsCopied(false);
      }, 2000); // Возвращаем исходный текст через 2 секунды
    } catch (err) {
      console.error("Ошибка при копировании текста: ", err);
      alert("Не удалось скопировать текст");
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
