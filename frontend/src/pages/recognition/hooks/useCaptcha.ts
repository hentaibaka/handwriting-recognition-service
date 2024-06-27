import { useState } from "react";

export const useCaptcha = () => {
  const [captcha, setCaptcha] = useState<string | null>(null);

  const generateCaptcha = () => {
    let uniqueChar = "";

    const randomChar =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (let i = 1; i < 5; i++) {
      uniqueChar += randomChar.charAt(Math.random() * randomChar.length);
    }

    setCaptcha(uniqueChar);
  };

  const checkCaptchaMatch = (value: string) => value === captcha;

  return {
    captcha,
    generateCaptcha,
    checkCaptchaMatch,
  };
};
