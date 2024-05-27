import { cn } from "@/lib/helpers";
import React, {
  ReactNode,
  DragEvent,
  useEffect,
  useState,
  useRef,
} from "react";

interface DropAndDropZoneProps {
  children: ReactNode;
  id?: string;
  className?: string;
  multiple?: boolean;
  onDragStateChange?: (isDragActive: boolean) => void;
  onDrag?: () => void;
  onDragIn?: () => void;
  onDragOut?: () => void;
  onDrop?: () => void;
  onClick?: () => void;
  onFilesDrop?: (file: File[]) => void;
}

export const DropAndDropZone = ({
  children,
  id,
  className,
  multiple = false,
  onFilesDrop,
  onDrag,
  onDrop,
  onDragIn,
  onDragOut,
  onClick,
  onDragStateChange,
}: DropAndDropZoneProps) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const mapFileListToArray = (files: FileList) => Array.from(files);

  const handleDragIn = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onDragIn?.();

    if (event.dataTransfer?.items && event.dataTransfer.items.length > 0) {
      setIsDragActive(true);
    }
  };

  const handleDragOut = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onDragOut?.();

    setIsDragActive(false);
  };

  const handleDrag = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    onDrag?.();
    if (!isDragActive) {
      setIsDragActive(true);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    setIsDragActive(false);
    onDrop?.();

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      const files = mapFileListToArray(event.dataTransfer.files);

      onFilesDrop?.(files);
      event.dataTransfer.clearData();
    }
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const files = mapFileListToArray(event.target.files);
      onFilesDrop?.(files);

      if (inputRef.current) {
        inputRef.current.value = "";
      }
    }
  };

  const handleClick = () => {
    inputRef.current?.click();
    onClick?.();
  };

  useEffect(() => {
    onDragStateChange?.(isDragActive);
  }, [isDragActive, onDragStateChange]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.code !== "Tab") e.preventDefault();
    if (e.code === "Enter" || e.code === "Space") {
      inputRef.current?.click();
    }
  };

  return (
    <div
      className={cn(
        "p-2 bg-white rounded-[10px] border-2 border-accent border-dashed",
        "flex justify-between items-center cursor-pointer duration-200",
        "focus-visible:ring ring-accent ring-offset-2",
        isDragActive && 'bg-accent/10',
        className
      )}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onDrag={(e) => e.stopPropagation()}
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
    >
      {children}

      <input
        id={id}
        ref={inputRef}
        className="hidden"
        type="file"
        accept="image/*"
        multiple={multiple}
        onChange={handleInputChange}
      />
    </div>
  );
};
