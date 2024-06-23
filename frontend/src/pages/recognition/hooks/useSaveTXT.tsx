import { useState } from "react";
import { DemodocsService } from "@/api";

export const useSaveTXT = () => {
  const [isLoading, setIsLoading] = useState(false);

  const getTxtBlobData = async (pageId: number) => {
    setIsLoading(true);
    
    const data = await DemodocsService.demodocsTextFileRetrieve(pageId);
    
    setIsLoading(false);
    
    return new Blob([data]);
  };

  return {
    isLoading,
    getTxtBlobData,
  };
};
