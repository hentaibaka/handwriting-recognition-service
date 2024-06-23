import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, DropAndDropZone, Spinner } from "@/components";
import { ComponentWithClassName } from "@/types";
import { Routes } from "@/lib/consts";
//import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/helpers";

export const DragAndDrop = ({ className }: ComponentWithClassName) => {
  //const { isAuth } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (files: FileList | File[]) => {
    if (!files[0]) return;

    //if (!isAuth) return navigate(Routes.signUp);

    const filteredFiles: File[] = [];

    for (let i = 0; i < files.length; i++) {
      if (files[i].type.includes("image")) filteredFiles.push(files[i]);
    }

    try {
      setIsLoading(true);
      navigate(
        { pathname: Routes.recognition },
        { state: { recognitionFiles: filteredFiles } }
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DropAndDropZone
      multiple
      className={cn(
        "h-[274px] px-[12px] flex flex-col justify-center items-center",
        className
      )}
      onFilesDrop={handleInputChange}
    >
      {isLoading ? (
        <Spinner className="h-16 w-16 border-4" />
      ) : (
        <>
          <Button className="flex flex-wrap h-[60px] px-[10px] md:h-[95px] md:px-[46px] md:py-[34px] mb-[12px]">
            <span className="font-semibold text-[1rem] md:text-[1.25rem] leading-[24px] mr-2">
              Распознать свой файл
            </span>
            <span className="font-semibold uppercase text-[0.7rem] md:text-[0.875rem] leading-[16px]">
              (Jpeg, png, tiff)
            </span>
          </Button>
          <span className="text-additional-color-3 hidden lg:block">
            или перетащите файл сюда
          </span>
        </>
      )}
    </DropAndDropZone>
  );
};
