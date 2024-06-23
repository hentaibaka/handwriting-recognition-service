import { PageSection } from "@/types";
import { cn } from "@/lib/helpers";

interface PageSectionsProps {
  className?: string;
  sections: PageSection[];
}

export const PageSections = ({ className, sections }: PageSectionsProps) => {
  const currentSectionName = "Статистика";

  return (
    <div
      className={cn(
        "h-[51px] w-fit px-[6px] sm:px-[12px] md:px-[24px] bg-additional-color-1 rounded-[10px] flex items-center gap-[12px] sm:gap-[16px] md:gap-[41px]",
        "text-[0.7rem] sm:text-[0.8rem] md:text-[1rem] lg:text-[1.25rem]",
        "font-medium md:font-normal",
        className
      )}
    >
      {sections.map(({ sectionId, sectionName }) => (
        <a
          key={sectionId}
          href={`#${sectionId}`}
          className={cn(currentSectionName === sectionName && "text-accent")}
        >
          {sectionName}
        </a>
      ))}
    </div>
  );
};
