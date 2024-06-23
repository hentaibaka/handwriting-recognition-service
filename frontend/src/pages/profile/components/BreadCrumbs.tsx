import { cn } from "@/lib/helpers";
import React from "react";
import { Link } from "react-router-dom";

interface BreadCrumbsProps extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
  currentPath: string;
  links: { path: string; label: string }[];
}

export const BreadCrumbs = ({
  className,
  currentPath,
  links,
}: BreadCrumbsProps) => (
  <div
    className={cn(
      "font-deja-vu-sans font-extralight text-additional-color-2 text-[0.8rem] sm:text-[0.95rem] leading-[1.1rem]",
      "flex gap-[12px]",
      className
    )}
  >
    {links.map(({ label, path }, index) => (
      <React.Fragment key={path}>
        <Link
          to={{ pathname: path }}
          className={cn(currentPath === path && "text-accent font-normal")}
        >
          {label}
        </Link>

        {links.length - 1 !== index && (
          <span className="font-inter font-light text-[0.8rem] leading-[0.96rem]">
            {">"}
          </span>
        )}
      </React.Fragment>
    ))}
  </div>
);
