import { Container } from "@/layouts/main-layout/components";
import {
  mainPageSections,
  projectDescriptionSectionId,
  projectObjectivesSectionId,
  relatedSectionId,
  statisticsSectionId,
} from "@/lib/consts";
import {
  DragAndDrop,
  PageSections,
  PageTitle,
  ProjectDescription,
  ProjectObjectives,
  Statistics,
  Related,
  UseSystem,
} from "./components";

export const MainPage = () => {
  return (
    <div className="pt-[200px] lg:pt-[236px]">
      <FixedContainer>
        <PageSections sections={mainPageSections} />
      </FixedContainer>

      <PageTitle className="mb-[28px]" />

      <DragAndDrop className="mb-[40px]" />

      <Statistics
        id={statisticsSectionId}
        className="mb-[113px] scroll-mt-[120px] sm:scroll-mt-[150px]"
      />

      <ProjectObjectives
        id={projectObjectivesSectionId}
        className="mb-[113px] scroll-mt-[150px] sm:scroll-mt-[195px]"
      />

      <ProjectDescription
        id={projectDescriptionSectionId}
        className="mb-[113px] scroll-mt-[150px] sm:scroll-mt-[190px] lg:scroll-mt-[120px]"
      />

      <Related
        id={relatedSectionId}
        className="mb-[113px] scroll-mt-[150px] sm:scroll-mt-[190px] lg:scroll-mt-[120px]"
      />

      <UseSystem className="mb-[47px]" />
    </div>
  );
};

const FixedContainer = ({ children }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className="fixed top-[80px] xl:top-[90px] left-0 right-0 z-[1000] sm:pt-[36px] bg-white">
    <Container>{children}</Container>
  </div>
);
