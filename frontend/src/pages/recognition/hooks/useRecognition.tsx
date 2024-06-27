import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { useSendFileForRecognition } from "./useSendFileForRecognition";

export interface RecognitionFileType {
  id: number | null;
  name?: string;
  imageUrl: string;
  text: string;
}

export const useRecognition = () => {
  const [selectedRecognitionFile, setSelectedRecognitionFile] =
    useState<RecognitionFileType | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const location = useLocation();
  const locationState: { recognitionFiles: File[] } = location.state;
  const [isShowCaptchaModal, setIsShowCaptchaModal] = useState(false);

  const openCaptchaModal = () => setIsShowCaptchaModal(true);
  const closeCaptchaModal = () => setIsShowCaptchaModal(false);

  const { isRecognitionPending, sendFileForRecognition } =
    useSendFileForRecognition({
      onSuccess: ({ text }) => {
        if (!uploadedFile || !text) return;

        const uploadedFileUrl = URL.createObjectURL(uploadedFile);

        setSelectedRecognitionFile({
          id: null,
          imageUrl: uploadedFileUrl,
          text: text,
        });
      },
    });

  const handleFileUpload = async (file: File) => {
    setUploadedFile(file);
    openCaptchaModal();
  };

  const handleCaptchaSolve = () => {
    if (uploadedFile) {
      sendFileForRecognition({ image: uploadedFile });
      setIsShowCaptchaModal(false);
    }
  };

  useEffect(() => {
    if (locationState) {
      handleFileUpload(locationState.recognitionFiles[0]);
      window.history.replaceState({}, "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    selectedRecognitionFile,
    setSelectedRecognitionFile,
    isRecognitionPending,
    handleFileUpload,
    captcha: {
      isShowCaptchaModal,
      closeCaptchaModal,
      handleCaptchaSolve,
    },
  };
};
