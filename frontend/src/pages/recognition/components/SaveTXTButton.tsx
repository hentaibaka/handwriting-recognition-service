import { Button } from "@/components";
import { useSaveTXT } from "../hooks";

interface SaveTXTButtonProps {
  fileId: number | null;
}

export const SaveTXTButton = ({ fileId }: SaveTXTButtonProps) => {
  const { isLoading, getTxtBlobData } = useSaveTXT();

  const handleSaveTxt = async () => {
    if (!fileId) return;

    const blobData = await getTxtBlobData(fileId);
    if (!blobData) return;

    const url = window.URL.createObjectURL(blobData);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute(
      "download",
      `Распознанный-текст-${new Date().toLocaleString()}.txt`
    );
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <Button
      className="font-deja-vu-sans"
      disabled={isLoading || !fileId}
      onClick={handleSaveTxt}
    >
      Сохранить в txt
    </Button>
  );
};
