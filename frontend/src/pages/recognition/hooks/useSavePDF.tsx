import { useState } from "react";
import { DemodocsService } from "@/api";

export const useSavePDF = () => {
  const [isLoading, setIsLoading] = useState(false);

  const getPdfBlobData = async (pageId: number) => {
    setIsLoading(true);
    
    const data = await DemodocsService.demodocsPdfFileRetrieve(pageId);
    
    setIsLoading(false);
    
    return new Blob([data]);
  };

  return {
    isLoading,
    getPdfBlobData,
  };
};
