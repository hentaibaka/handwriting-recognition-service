import React, { ReactNode } from "react";
import { Button, Card, Icon } from "@/components";
import { cn } from "@/lib/helpers";
import { DemoDocsDto } from "@/api";
import "./RecognitionFilesSlider.scss";
import { RecognitionFileType, useRecognitionFilesSlider } from "../../hooks";

interface RecognitionFilesSliderProps
  extends React.HTMLAttributes<HTMLDivElement> {
  buttonSlot?: ReactNode;
  demoDocs?: DemoDocsDto[];
  isLoading: boolean;
  selectedFile: RecognitionFileType | null;
  setSelectedFile: (file: RecognitionFileType) => void;
}

export const RecognitionFilesSlider = ({
  className,
  buttonSlot,
  demoDocs = [],
  isLoading,
  selectedFile,
  setSelectedFile,
  ...props
}: RecognitionFilesSliderProps) => {
  const {
    containerRef,
    firstElementRef,
    lastElementRef,
    isDisableSlideLeft,
    isDisableSlideRight,
    elementsGap,
    handleSlideLeft,
    handleSlideRight,
  } = useRecognitionFilesSlider();

  const handleSelectRecognitionFile = (recognitionFileName: string) => {
    const recognitionFile = demoDocs.find(
      ({ name }) => name === recognitionFileName
    );

    if (recognitionFile)
      setSelectedFile({
        name: recognitionFile.name,
        imageUrl: `${import.meta.env.VITE_API_URL}${recognitionFile.url}`,
        text: recognitionFile.text,
      });
  };

  const getRecognitionFileRef = (index: number) =>
    index === 0
      ? firstElementRef
      : index === demoDocs.length - 1
      ? lastElementRef
      : undefined;

  return (
    <div className={cn("recognition-slider", className)} {...props}>
      <div
        ref={containerRef}
        className="recognition-slide-list pt-[5px] pb-[25px] px-[5px] gap-[10px]"
        style={{ gap: elementsGap }}
      >
        {!isLoading ? (
          demoDocs?.map(({ name, url }, index) => (
            <RecognitionFile
              key={name}
              ref={getRecognitionFileRef(index)}
              image={`${import.meta.env.VITE_API_URL}${url}`}
              title={name}
              isSelected={selectedFile?.name === name}
              onClick={() => handleSelectRecognitionFile(name)}
            />
          ))
        ) : (
          <FilesSkeleton />
        )}
      </div>

      <div className="flex flex-wrap gap-[20px] ">
        {buttonSlot}

        <div className="flex gap-[7px]">
          <Button
            className="max-h-[53px] w-[53px] font-inter"
            disabled={isDisableSlideLeft || isLoading}
            onClick={handleSlideLeft}
          >
            <Icon iconName="caretLeft" />
          </Button>

          <Button
            className="max-h-[53px] w-[53px] font-inter"
            disabled={isDisableSlideRight || isLoading}
            onClick={handleSlideRight}
          >
            <Icon iconName="caretRight" />
          </Button>
        </div>
      </div>
    </div>
  );
};

interface RecognitionFileProps extends React.HTMLAttributes<HTMLDivElement> {
  image: string;
  title: string;
  isSelected: boolean;
}

const RecognitionFile = React.forwardRef<HTMLDivElement, RecognitionFileProps>(
  ({ image, title, isSelected, ...props }, ref) => (
    <div
      ref={ref}
      className="recognition-slide py-[2px] px-[4px] w-[188px]"
      {...props}
    >
      <Card
        className={cn(
          "p-[10px] h-full rounded-[10px] hover:ring hover:ring-accent/70 duration-100 select-none",
          isSelected && "ring ring-accent/40"
        )}
      >
        <div className="w-[160px]">
          <img
            className="mb-[13px] h-[210px] w-full object-fill pointer-events-none"
            src={image}
            alt={title}
          />
          <div className="text-accent text-[0.75rem] leading-[1rem]">
            {title}
          </div>
        </div>
      </Card>
    </div>
  )
);

const FilesSkeleton = () => (
  <>
    <div className="min-w-[188px] h-[259px] py-[2px] px-[4px]">
      <Card className="rounded-[10px] h-full animate-pulse bg-neutral-100"></Card>
    </div>
    <div className="min-w-[188px] h-[259px] py-[2px] px-[4px]">
      <Card className="rounded-[10px] h-full animate-pulse bg-neutral-100"></Card>
    </div>
    <div className="min-w-[188px] h-[259px] py-[2px] px-[4px]">
      <Card className="rounded-[10px] h-full animate-pulse bg-neutral-100"></Card>
    </div>
    <div className="min-w-[188px] h-[259px] py-[2px] px-[4px]">
      <Card className="rounded-[10px] h-full animate-pulse bg-neutral-100"></Card>
    </div>
    <div className="min-w-[188px] h-[259px] py-[2px] px-[4px]">
      <Card className="rounded-[10px] h-full animate-pulse bg-neutral-100"></Card>
    </div>
    <div className="min-w-[188px] h-[259px] py-[2px] px-[4px]">
      <Card className="rounded-[10px] h-full animate-pulse bg-neutral-100"></Card>
    </div>
  </>
);
