import { RecognizeImageDto, RecognizeService } from "@/api";
import { useMutation } from "@tanstack/react-query";

interface Props {
  onSuccess?: (
    data: { text: string },
    variables: RecognizeImageDto,
    context: unknown
  ) => unknown;
}

export const useSendFileForRecognition = ({ onSuccess }: Props) => {
  const { isPending, mutate } = useMutation({
    mutationFn: (file: RecognizeImageDto) =>
      RecognizeService.recognizeCreate(file),
    onSuccess: onSuccess,
  });

  return {
    sendFileForRecognition: mutate,
    isRecognitionPending: isPending,
  };
};
