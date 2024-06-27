import { useEffect, useState } from "react";
import { Dialog, DialogContent } from "@/components";
import { useCaptcha } from "../hooks/useCaptcha";

interface CaptchaModalProps {
  isShow: boolean;
  onClose: () => void;
  onSuccessfulSolvedCaptcha: () => void;
}

export const CaptchaModal = ({
  isShow,
  onClose,
  onSuccessfulSolvedCaptcha,
}: CaptchaModalProps) => {
  const { captcha, generateCaptcha, checkCaptchaMatch } = useCaptcha();
  const [value, setValue] = useState("");

  const handleOpenChange = (value: boolean) => {
    if (!value) onClose();
  };

  const handleButtonClick = () => {
    if (checkCaptchaMatch(value)) onSuccessfulSolvedCaptcha();
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) =>
    setValue(event.target.value);

  useEffect(() => {
    generateCaptcha();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Dialog open={isShow} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-[708px]">
        <div className="flex items-center gap-5">
          <div id="user-input" className="inline">
            <input
              className="input"
              type="text"
              placeholder="Captcha code"
              value={value}
              onChange={handleInputChange}
            />
          </div>

          <div className="inline cursor-pointer" onClick={generateCaptcha}>
            Сгенерировать
          </div>

          <div id="image" className="inline">
            {captcha}
          </div>
          <button id="btn" className="input" onClick={handleButtonClick}>
            Отправить
          </button>

          <p id="key"></p>
        </div>
      </DialogContent>
    </Dialog>
  );
};
