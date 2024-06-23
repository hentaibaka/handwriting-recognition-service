export interface ComponentWithClassName {
  className?: string;
}

export interface PageSection {
  sectionId: string;
  sectionName: string;
}

export interface RequestError extends Error {
  body?: Record<string, string>;
}
