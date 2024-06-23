import { Icon } from "@/components";
import { Container } from "./Container";
import { cn } from "@/lib/helpers";
import { footerEmail } from "@/lib/consts";
import IkitLogo from "@/assets/images/logo252.png";

export const Footer = () => {
  return (
    <footer className="min-h-[118px] sm:px-[12px] pt-[24px] pb-[14px] text-white bg-additional-color-6 overflow-hidden">
      <Container>
        <div className="grid gap-[16px] sm:gap-[12px] place-items-center sm:place-items-start sm:grid-cols-2 xl:grid-cols-4">
          <div className="min-w-[170px] h-[41px]">

          </div>

          <div className="flex flex-wrap sm:flex-nowrap justify-center items-center sm:w-[296px] text-center sm:text-left text-[0.8rem] md:text-[0.875rem] md:leading-[1.125rem]">
            <a href="https://govreport.sfu-kras.ru/" target="_blank" >ОТЧЁТЫ ГУБЕРНАТОРОВ</a>
          </div>
          <div className="flex flex-wrap sm:flex-nowrap justify-center items-center sm:w-[296px] text-center sm:text-left text-[0.8rem] md:text-[0.875rem] md:leading-[1.125rem]">
            <a href="https://fromthepage.sfu-kras.ru/lib/governors-reports" target="_blank" >FROM THE PAGE</a>
          </div>

          <div className="min-w-[170px] h-[41px]">

          </div>

          <Icon
            iconName="logo-footer"
            className={cn(
              "relative -left-1.5",
              "min-w-[200px] h-[34px] sm:min-w-[240px] sm:h-[40px] md:min-w-[260px] md:min-h-[46px]"
            )}
          />
          <Icon
            iconName="sibFU-logo"
            className={cn("text-white", "min-w-[170px] h-[41px]")}
          />
          <img
                className="min-w-[170px] h-[41px] bg-contain"
                src={IkitLogo}
                alt="IKIT-logo"
              />
          <div className="flex flex-wrap sm:flex-nowrap justify-center items-center gap-[8px] text-[0.8rem] md:text-[0.875rem]">
            <span className="leading-tight">Остались вопросы?<br/>Пишите нам!</span>
            <div className="flex items-center gap-[9px]">
              <Icon iconName="email" />
              <a href={`mailto:${footerEmail}`} className="leading-tight">
                {footerEmail}
              </a>
            </div>
          </div>
          <div className="sm:w-[257px] text-center sm:text-left text-white/60 text-[0.7rem] md:text-[0.75rem] leading-[1rem]">
            © Сибирский федеральный университет, 2006-{new Date().getFullYear()}
          </div>
          <div className="sm:w-[296px] text-center sm:text-left text-[0.8rem] md:text-[0.875rem] md:leading-[1.125rem]">
            Свободный пр., 82А, Красноярск, Красноярский край, 660041
          </div>
          <div className="sm:w-[296px] text-center sm:text-left text-[0.8rem] md:text-[0.875rem] md:leading-[1.125rem]">
            ул. Академика Киренского, 26Б, Красноярск, Красноярский край, 660074
          </div>
          <div className="sm:w-[300px] text-center sm:text-left text-white/60 text-[0.7rem] md:text-[0.75rem] leading-[1rem]">
            При использовании текстовых и графических материалов ссылка на сайт
            обязательна
          </div>
        </div>
      </Container>
    </footer>
  );
};
