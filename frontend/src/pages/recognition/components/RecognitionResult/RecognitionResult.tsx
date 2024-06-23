import { ReactNode, useEffect, useRef, useState } from "react";
import { Card } from "@/components";
import { cn } from "@/lib/helpers";

interface RecognitionResultProps extends React.HTMLAttributes<HTMLDivElement> {
  fileImage?: string;
  text?: string;
  buttonsSlot: ReactNode;
}

export const RecognitionResult = ({
  className,
  fileImage,
  text,
  buttonsSlot,
  ...props
}: RecognitionResultProps) => {
  const [textCardHeight, setTextCardHeight] = useState<number | null>(null);
  const imageContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setTimeout(
      () => setTextCardHeight(imageContainerRef.current?.clientHeight || null),
      300
    );
  }, [fileImage, text]);

  return (
    <div className={cn("grid lg:grid-cols-2 gap-[34px]", className)} {...props}>
      <div
        ref={imageContainerRef}
        className={cn(
          "h-fit rounded-[11px] grid place-items-center bg-additional-color-2",
          "p-[12px] sm:p-[22px] md:p-[40px] lg:p-[60px]"
        )}
      >
        {fileImage && (
          <img
            className="w-full object-contain"
            src={fileImage}
            alt="Выбранный документ"
          />
        )}
      </div>

      <Card
        className="p-[20px] md:py-[27px] md:px-[40px] flex flex-col"
        style={{ height: textCardHeight || undefined }}
      >
        <div
          className="mb-[9px] h-full overflow-auto flex-grow font-deja-vu-sans font-extralight text-[1.125rem] leading-[1.35rem]"
          dangerouslySetInnerHTML={{ __html: text || '' }}
        />

        <div className="flex justify-center gap-3">{buttonsSlot}</div>
      </Card>
    </div>
  );
};
