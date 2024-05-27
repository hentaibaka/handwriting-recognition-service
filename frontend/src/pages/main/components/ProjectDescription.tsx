import { Card } from "@/components";
import { cn } from "@/lib/helpers";
import projectDescriptionFile1 from "@/assets/images/projectDescriptionFile1.png";
import projectDescriptionFile2 from "@/assets/images/projectDescriptionFile2.png";
import projectDescriptionFile3 from "@/assets/images/projectDescriptionFile3.svg";

interface ProjectDescriptionProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export const ProjectDescription = ({
  className,
  ...props
}: ProjectDescriptionProps) => {
  return (
    <div className={cn(className)} {...props}>
      <div className="grid items-center gap-[20px] sm:gap-[32px] lg:grid-cols-[1fr_2fr]">
        <div>
          <h3
            className={cn(
              "mb-[30px] w-fit bg-additional-color-1 rounded-[10px] uppercase font-semibold text-accent",
              "text-[1.5rem] md:text-[1.875rem] md:leading-[28px]",
              "p-[10px] md:p-[18px]"
            )}
          >
            Описание
          </h3>

          <div className="text-additional-color-3 font-medium text-[1.2rem] md:text-[1.625rem] md:leading-[1.625rem]">
            Больше чем просто распознавание текста
          </div>
        </div>

        <Card className="flex flex-col md:flex-row gap-[20px] md:gap-[52px] px-[6px] py-[8px] sm:px-[20px] sm:py-[22px] md:px-[45px] md:py-[43px]">
          <div>
            <div className="md:mt-[30px] mb-[11px] font-medium text-accent md:text-[1.125rem] leading-[1.125rem]">
              Для библиотекарей
            </div>
            <div className="mb-[30px] text-additional-color-3 font-medium md:text-[1.25rem] leading-[1.25rem]">
              Задача проекта - распознавание рукописных текстов на исторических
              документах
            </div>
            <div className="text-additional-color-5 md:text-[1.125rem] md:leading-[1.5rem]">
              Создание программы для автоматического распознавания рукописного
              текста
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-[10px]">
            <Card className="p-[10px] flex flex-col items-center w-fit h-fit rounded-[7px]">
              <img
                className="mb-[10px] h-[150px] min-w-[116px] bg-contain"
                src={projectDescriptionFile1}
                alt="file1"
              />
              <div className="text-accent text-[0.6rem] leading-[0.687rem]">
                Отчет губернатора за 1852 г.
              </div>
            </Card>
            <Card className="p-[10px] flex flex-col items-center w-fit h-fit rounded-[7px]">
              <img
                className="mb-[10px] h-[150px] min-w-[116px] bg-contain"
                src={projectDescriptionFile2}
                alt="file2"
              />
              <div className="text-accent text-[0.6rem] leading-[0.687rem]">
                Отчет губернатора за 1852 г.
              </div>
            </Card>
          </div>
        </Card>

        <Card className="p-[12px] sm:px-[20px] sm:py-[22px] md:px-[45px] md:py-[43px]">
          <div className="mb-[14px] md:mb-[28px] text-additional-color-3 font-medium text-[0.8rem] sm:text-[1.1rem] md:text-[1.25rem] md:leading-[1.25rem]">
            Распознавание рукописного текста с помощью искусственного интеллекта
            и библиотеки Tesseract - OCR
          </div>

          <div className="text-additional-color-5 text-[0.8rem] sm:text-[1rem] md:text-[1.125rem] md:leading-[1.5rem]">
            ПО OCR очищает изображение и удаляет ошибочные области
          </div>
        </Card>

        <Card
          className="bg-[160%_35%] bg-[length:55%] bg-no-repeat p-[12px] sm:px-[20px] sm:py-[22px] md:px-[45px] md:py-[43px]"
          style={{ backgroundImage: `url(${projectDescriptionFile3})` }}
        >
          <div className="max-w-[75%] md:max-w-[550px]">
            <div className="mb-[14px] md:mb-[24px] text-additional-color-3 font-medium text-[0.8rem] sm:text-[1.1rem] md:text-[1.25rem] md:leading-[1.25rem]">
              Создание веб-интерфейса и пакетная загрузка изображений текста
            </div>

            <div className="font-deja-vu-sans text-additional-color-5 text-[0.8rem] sm:text-[1rem] md:text-[1.125rem] md:leading-[1.5rem]">
              Реализация веб-интерфейса для работы с программой. Возможность
              пакетной загрузки изображений рукописного текста в формате:{" "}
              <span className="text-accent font-bold text-[0.7rem] sm:text-[1rem] md:text-[1.125rem] md:leading-[1.5rem]">
                TIFF, JPEG, PNG
              </span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
