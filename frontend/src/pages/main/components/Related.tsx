import { cn } from "@/lib/helpers";

interface RelatedProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export const Related = ({
  className,
  ...props
}: RelatedProps) => {
  return (
    <div className={cn(className)} {...props}>
      <div
        className={cn(
          "mb-[30px] w-fit bg-additional-color-1 rounded-[10px] uppercase font-semibold text-accent ",
          "text-[1.5rem] md:text-[1.875rem] md:leading-[28px]",
          "p-[10px] md:p-[18px]"
        )}
      >
        Связанное
      </div>

      <div className="flex flex-col sm:flex-row gap-[26px] md:gap-[50px] lg:gap-[76px]">
        <div className="flex flex-col md:flex-row">
            <div>
              <div className="hover:underline md:mt-[30px] mb-[11px] font-deja-vu-sans font-bold text-accent md:text-[1.125rem] leading-[1.125rem]">
                <a rel="noopener" href="https://govreport.sfu-kras.ru" target="_blank">
                  GovReport
                </a>
              </div>
              <div className="font-deja-vu-sans text-additional-color-5 text-[0.8rem] sm:text-[1rem] md:text-[1.125rem] md:leading-[1.5rem]">
                Обширный архив оцифрованных отчетов губернаторов Енисейской губернии с удобным доступом и поиском
              </div>
            </div>
          </div>
        
          <div className="flex flex-col md:flex-row">
            <div>
              <div className="hover:underline md:mt-[30px] mb-[11px] font-deja-vu-sans font-bold text-accent md:text-[1.125rem] leading-[1.125rem]">
                <a rel="noopener" href="https://fromthepage.sfu-kras.ru/lib/governors-reports" target="_blank">
                  FromThePage
                </a>
              </div>
              <div className="font-deja-vu-sans text-additional-color-5 text-[0.8rem] sm:text-[1rem] md:text-[1.125rem] md:leading-[1.5rem]">
                Платформа для расшифровки рукописных документов и улучшения их транскрипции через совместную работу пользователей
              </div>
            </div>
          </div>
      </div>
    </div>
  );
};
