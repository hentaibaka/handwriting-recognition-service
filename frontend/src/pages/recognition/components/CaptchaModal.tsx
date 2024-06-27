import { useEffect, useState } from "react";
import { Dialog, DialogContent } from "@/components";
import { useCaptcha } from "../hooks/useCaptcha";
import { Button } from "@/components";

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
    else generateCaptcha();
  };

  const handleButtonClick = () => {
    if (checkCaptchaMatch(value)) onSuccessfulSolvedCaptcha();
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) =>
    setValue(event.target.value);

  useEffect(() => {
    if (isShow) generateCaptcha();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isShow]);

  return (
    <Dialog open={isShow} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-[708px]">
        <div className="grid grid-cols-2 place-items-center items-center gap-5">
          <input
            className="input"
            type="text"
            placeholder="Captcha code"
            value={value}
            onChange={handleInputChange}
          />

          <div id="image">
            {captcha}
          </div>

          <Button
            className="input font-deja-vu-sans"
            onClick={handleButtonClick}
          >
            Отправить
          </Button>

          <Button
            className="font-deja-vu-sans"
            onClick={generateCaptcha}
          >
            Сгенерировать заново
          </Button>

          <p id="key"></p>
        </div>
      </DialogContent>
    </Dialog>
  );
};
