import { Button } from "@/components";
import { useSavePDF } from "../hooks/useSavePDF";

interface SavePDFButtonProps {
  fileId: number | null;
}

export const SavePDFButton = ({ fileId }: SavePDFButtonProps) => {
  const { isLoading, getPdfBlobData } = useSavePDF();

  const handleSavePdf = async () => {
    if (!fileId) return;

    const blobData = await getPdfBlobData(fileId);
    if (!blobData) return;

    const url = window.URL.createObjectURL(blobData);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute(
      "download",
      `Распознанный-текст-${new Date().toLocaleString()}.pdf`
    );
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <Button
      className="font-deja-vu-sans"
      disabled={isLoading || !fileId}
      onClick={handleSavePdf}
    >
      Сохранить в pdf
    </Button>
  );
};
