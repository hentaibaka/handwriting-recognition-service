import { useWindowDimensions } from "@/hooks";
import { useEffect, useRef, useState } from "react";

const fileElementWidth = 188;

export const useRecognitionFilesSlider = () => {
    const [elementsGap, setElementsGap] = useState<number | undefined>(undefined);
    const [isDisableSlideLeft, setIsDisableSlideLeft] = useState(false);
    const [isDisableSlideRight, setIsDisableSlideRight] = useState(false);
  
    const containerRef = useRef<HTMLDivElement | null>(null);
    const firstElementRef = useRef<HTMLDivElement | null>(null);
    const lastElementRef = useRef<HTMLDivElement | null>(null);
  
    const { width } = useWindowDimensions();
    const isDesktopScreen = width ? width >= 1024 : false;
  
    const handleSlideRight = () => {
      if (!containerRef.current) return;
      containerRef.current.style.scrollBehavior = "smooth";
      containerRef.current.scrollLeft += fileElementWidth;
    };
  
    const handleSlideLeft = () => {
      if (!containerRef.current) return;
      containerRef.current.style.scrollBehavior = "smooth";
      containerRef.current.scrollLeft -= fileElementWidth;
    };
  
    const calculateContainerGap = () => {
      if (!containerRef.current) return;
  
      const containerClientWidth = containerRef.current.clientWidth;
  
      const countFileElementsInContainer = Math.floor(
        containerClientWidth / fileElementWidth
      );
      const sumFileElementsWidth =
        countFileElementsInContainer * fileElementWidth;
      const containerClientWidthWithoutGap =
        containerClientWidth - sumFileElementsWidth;
  
      const countGapInContainerClientWidth =
        countFileElementsInContainer - 1 > 1
          ? countFileElementsInContainer - 1
          : countFileElementsInContainer;
  
      const gap = containerClientWidthWithoutGap / countGapInContainerClientWidth;
  
      setElementsGap(gap);
    };
  
    useEffect(() => {
      if (isDesktopScreen) calculateContainerGap();
      else setElementsGap(undefined);
    }, [width, isDesktopScreen]);
  
    useEffect(() => {
      const firstElementObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) setIsDisableSlideLeft(true);
            else setIsDisableSlideLeft(false);
          });
        },
        {
          root: containerRef.current,
          rootMargin: "100px 0px",
        }
      );
  
      if (firstElementRef.current)
        firstElementObserver.observe(firstElementRef.current);
  
      return () => {
        if (firstElementRef.current)
          // eslint-disable-next-line react-hooks/exhaustive-deps
          firstElementObserver.unobserve(firstElementRef.current);
      };
    }, [firstElementRef]);
  
    useEffect(() => {
      const lastElementObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) setIsDisableSlideRight(true);
            else setIsDisableSlideRight(false);
          });
        },
        {
          root: containerRef.current,
          rootMargin: "100px 0px",
          threshold: 1
        }
      );
  
      if (lastElementRef.current)
        lastElementObserver.observe(lastElementRef.current);
  
      return () => {
        if (lastElementRef.current)
          // eslint-disable-next-line react-hooks/exhaustive-deps
          lastElementObserver.unobserve(lastElementRef.current);
      };
    }, [lastElementRef]);

  return {
    containerRef,
    elementsGap,
    firstElementRef,
    lastElementRef,
    handleSlideLeft,
    handleSlideRight,
    isDisableSlideLeft,
    isDisableSlideRight
  }
}
