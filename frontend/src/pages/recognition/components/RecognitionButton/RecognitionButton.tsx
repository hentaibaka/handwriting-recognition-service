import { useRef } from "react";
import { Button } from "@/components";

interface RecognitionButtonProps {
  onFileUpload: (file: File) => void;
}

export const RecognitionButton = ({ onFileUpload }: RecognitionButtonProps) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const file = event.target.files[0];
      onFileUpload(file);

      if (inputRef.current) {
        inputRef.current.value = "";
      }
    }
  };

  const handleButtonClick = () => inputRef.current?.click();

  return (
    <>
      <Button className="max-h-[53px]" onClick={handleButtonClick}>
        Распознать
      </Button>
      <input
        ref={inputRef}
        className="hidden"
        type="file"
        accept="image/*"
        multiple={false}
        onChange={handleInputChange}
      />
    </>
  );
};
