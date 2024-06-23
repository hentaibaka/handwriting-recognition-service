import { PageSection } from "@/types";

export const isShowVkIDButton = false;

export const footerEmail = "ai@sfu-kras.ru";

export const statisticsSectionId = "statistics";
export const projectObjectivesSectionId = "project-objectives";
export const projectDescriptionSectionId = "project-description";

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
];
