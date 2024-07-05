
import { useEffect, useState } from "react";
import { Dialog, DialogContent } from "@/components";
import { Button } from "@/components";
import ReCAPTCHA from "react-google-recaptcha";

interface CaptchaModalProps {
  isShow: boolean;
  onClose: () => void;
  onSuccessfulSolvedCaptcha: (token: string) => void;
}

export const CaptchaModal = ({
  isShow,
  onClose,
  onSuccessfulSolvedCaptcha,
}: CaptchaModalProps) => {
  const [recaptchaToken, setRecaptchaToken] = useState<string | null>(null);

  const handleOpenChange = (value: boolean) => {
    if (!value) onClose();
  };

  const handleRecaptchaChange = (token: string | null) => {
    setRecaptchaToken(token);
  };

  const handleButtonClick = () => {
    if (recaptchaToken) {
      onSuccessfulSolvedCaptcha(recaptchaToken);
    } else {
      alert("Please solve the reCAPTCHA");
    }
  };

  useEffect(() => {
    if (isShow) {
      setRecaptchaToken(null);
    }
  }, [isShow]);

  return (
    <Dialog open={isShow} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-[376px]">
        <div className="grid place-items-center items-center gap-5">
          <ReCAPTCHA
            sitekey="6Ldc1gcqAAAAAA7p_yj-vnvytitBFBsQpj7Sk2wg"
            onChange={handleRecaptchaChange}
          />
          <Button className="input font-deja-vu-sans" onClick={handleButtonClick}>
            Отправить
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
