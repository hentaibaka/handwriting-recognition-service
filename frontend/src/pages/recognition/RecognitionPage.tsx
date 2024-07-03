import { Card, Spinner } from "@/components";
import { cn } from "@/lib/helpers";
import {
  CaptchaModal,
  RecognitionButton,
  RecognitionFilesSlider,
  RecognitionResult,
  //SavePDFButton,
  SaveTXTButton,
} from "./components";
import { useDemoDocs, useRecognition } from "./hooks";
import { CopyTextButton } from "./components/CopyTextButton";
import axios from "axios";

export const RecognitionPage = () => {
  const {
    selectedRecognitionFile,
    setSelectedRecognitionFile,
    isRecognitionPending,
    handleFileUpload,
    captcha,
  } = useRecognition();
  const { demoDocs, isDemoDocsLoading } = useDemoDocs();

  const handleCaptchaSolve = async (token: string) => {
    function getCookie(name: string) {
      let cookieValue = null;
      if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
          const cookie = cookies[i].trim();
          // Проверяем, начинается ли эта строка куки с нужного нам имени
          if (cookie.substring(0, name.length + 1) === (name + '=')) {
            cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
            break;
          }
        }
      }
      return cookieValue;
    }
    try {
      const csrftoken = getCookie('csrftoken'); // Получение CSRF токена из куки
      const response = await axios.post(
        'api/submit-form/',
        { recaptchaToken: token },
        {
          headers: {
            'X-CSRFToken': csrftoken, // Добавление заголовка X-CSRFToken
          },
          withCredentials: true, // Включение передачи куки
        }
      );

      if (response.status === 200) {
        // Handle successful captcha solve
        captcha.closeCaptchaModal();
        // Вызывайте другие методы или обновляйте состояние здесь
        captcha.handleCaptchaSolve();
      } else {
        // Handle captcha solve failure
        alert('Captcha verification failed, please try again.');
      }
    } catch (error) {
      console.error('Error verifying captcha:', error);
    }
  };

  return (
    <div className="px-[6px]">
      <PageHeader />

      <Card className="px-[15px] py-[30px] mb-[20px]">
        <RecognitionFilesSlider
          isLoading={isDemoDocsLoading}
          selectedFile={selectedRecognitionFile}
          setSelectedFile={setSelectedRecognitionFile}
          demoDocs={demoDocs}
          buttonSlot={
            <>
              <RecognitionButton onFileUpload={handleFileUpload} />
            </>
          }
        />
      </Card>

      {isRecognitionPending ? (
        <div className="h-[250px] flex justify-center items-center">
          <Spinner className="!w-16 !h-16 border-4" />
        </div>
      ) : (
        <>
          {selectedRecognitionFile ? (
            <RecognitionResult
              className="mb-[36px]"
              fileImage={selectedRecognitionFile?.imageUrl}
              text={selectedRecognitionFile?.text}
              buttonsSlot={
                <>
                  <SaveTXTButton fileId={selectedRecognitionFile.id} />
                  <CopyTextButton text={selectedRecognitionFile.text} />
                </>
              }
            />
          ) : (
            <div className="h-[100px] flex justify-center items-center text-[1.2rem]">
              Выберите файл для распознавания
            </div>
          )}
        </>
      )}

      <CaptchaModal
        isShow={captcha.isShowCaptchaModal}
        onClose={captcha.closeCaptchaModal}
        onSuccessfulSolvedCaptcha={handleCaptchaSolve}
      />
    </div>
  );
};

const PageHeader = () => (
  <div className="mt-[10px] mb-[51px] lg:w-[800px] mx-auto">
    <h2
      className={cn(
        "mb-[10px] uppercase text-center text-additional-color-3 font-medium ",
        "sm:text-[1.2rem] lg:text-[1.875rem] lg:leading-[2.5rem]"
      )}
    >
      Сервис преобразования рукописного текста в{" "}
      <span className="font-semibold text-accent">цифровой формат</span>
    </h2>

    <h3
      className={cn(
        "font-deja-vu-sans font-normal text-additional-color-7 text-center",
        "md:text-[1.2rem] lg:text-[1.5rem] lg:leading-[1.75rem]"
      )}
    >
      Выберите файлы для сканирования
    </h3>
  </div>
);