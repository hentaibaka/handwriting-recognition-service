import { PageSection } from "@/types";

export const isShowVkIDButton = true;

export const footerEmail = "manuscript@sfu-kras.ru";

export const statisticsSectionId = "statistics";
export const projectObjectivesSectionId = "project-objectives";
export const projectDescriptionSectionId = "project-description";
export const relatedSectionId = "related";

export const mainPageSections: PageSection[] = [
  {
    sectionId: statisticsSectionId,
    sectionName: "Статистика",
  },
  {
    sectionId: projectObjectivesSectionId,
    sectionName: "Цель проекта",
  },
  {
    sectionId: projectDescriptionSectionId,
    sectionName: "Описание проекта",
  },
  {
    sectionId: relatedSectionId,
    sectionName: "Связанное",
  },
];
