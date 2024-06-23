import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import * as Portal from "@radix-ui/react-portal";
import { Button, Icon } from "@/components";
import { useAuth } from "@/lib/auth";
import { mainPageSections, Routes } from "@/lib/consts";
import { cn } from "@/lib/helpers";
import { Container } from "./Container";

export const Header = () => {
  const { isAuth } = useAuth();
  const location = useLocation();
  const [isNavbarOpen, setIsNavbarOpen] = useState(false);
  const headerRef = useRef<HTMLElement | null>(null);
  const isMainPage = location.pathname === Routes.main;

  const navbarToggleHandler = () => {
    setIsNavbarOpen(!isNavbarOpen);
    if (!isNavbarOpen) document.body.style.overflow = "hidden";
    else document.body.removeAttribute("style");
  };

  const handleClickOutsideHeader = (event: MouseEvent) => {
    if (
      headerRef.current &&
      !headerRef.current.contains(event.target as Node)
    ) {
      setIsNavbarOpen(false);
      document.body.removeAttribute("style");
    }
  };

  useEffect(() => {
    window.addEventListener("click", handleClickOutsideHeader);
    return () => {
      window.removeEventListener("click", handleClickOutsideHeader);
    };
  }, []);

  return (
    <Container
      className={cn(
        (isMainPage || isNavbarOpen) && "z-[1000] fixed top-0 left-0 right-0"
      )}
    >
      <header
        ref={headerRef}
        className={cn(
          "h-[90px] bg-white w-full flex justify-between sm:justify-start items-center mb-[25px] relative"
        )}
      >
        <Link to={{ pathname: Routes.main }}>
          <Icon
            iconName="logo"
            className="w-[80%] h-full sm:min-w-[180px] sm:h-[30px] md:min-w-[250px] md:h-[40px] lg:min-w-[283px] lg:h-[50px]"
          />
        </Link>

        <div className="hidden sm:block ml-[20px] mr-[16px] md:ml-[39px] md:mr-[22px] w-full h-[2px] bg-accent" />

        <div className="hidden md:block">
          {isAuth ? (
            <div className="flex gap-[27px]">
              <Link to={{ pathname: Routes.recognition }} tabIndex={-1}>
                <Button variant="ghost" className="uppercase">
                  Демонстрация
                </Button>
              </Link>

              <Link to={{ pathname: Routes.profile }}>
                <Button variant="ghost" className="uppercase">
                  Личный кабинет
                </Button>
              </Link>
            </div>
          ) : (
            <div className="flex gap-[27px]">
              <Link to={{ pathname: Routes.recognition }} tabIndex={-1}>
                <Button variant="ghost" className="uppercase">
                  Демонстрация
                </Button>
              </Link>

              <Link to={{ pathname: Routes.signUp }} tabIndex={-1}>
                <Button variant="ghost" className="uppercase">
                  Регистрация
                </Button>
              </Link>

              <Link to={{ pathname: Routes.signIn }} tabIndex={-1}>
                <Button className="uppercase">Авторизация</Button>
              </Link>
            </div>
          )}
        </div>

        <div className="md:hidden">
          <MobileNavbarButton
            isNavbarOpen={isNavbarOpen}
            onClick={navbarToggleHandler}
          />

          <MobileNavbar isNavbarOpen={isNavbarOpen} isAuth={isAuth} />
        </div>
      </header>
    </Container>
  );
};

interface MobileNavbarButtonProps {
  isNavbarOpen: boolean;
  onClick: () => void;
}

const MobileNavbarButton = ({
  isNavbarOpen,
  onClick,
}: MobileNavbarButtonProps) => (
  <button
    onClick={onClick}
    aria-label="Mobile Menu"
    className={cn("block rounded-lg px-3 py-[6px] ring-accent focus:ring")}
  >
    <span
      className={cn(
        "bg-accent relative my-1.5 block h-0.5 w-[30px] transition-all duration-300",
        isNavbarOpen && "top-[7px] rotate-45"
      )}
    />
    <span
      className={cn(
        "bg-accent relative my-1.5 block h-0.5 w-[30px] transition-all duration-300",
        isNavbarOpen && "opacity-0"
      )}
    />
    <span
      className={cn(
        "bg-accent relative my-1.5 block h-0.5 w-[30px] transition-all duration-300",
        isNavbarOpen && "top-[-8.5px] -rotate-45"
      )}
    />
  </button>
);

interface MobileNavbarProps {
  isNavbarOpen: boolean;
  isAuth: boolean;
}

const MobileNavbar = ({ isNavbarOpen, isAuth }: MobileNavbarProps) => {
  const location = useLocation();
  const isMainPage = location.pathname === Routes.main;

  return (
    <Portal.Root>
      <nav
        className={cn(
          "z-[1100] fixed bottom-0 sm:bottom-auto right-0 left-0 sm:left-auto duration-200",
          "bg-white border rounded-[10px] py-[20px] px-[30px]",
          isNavbarOpen
            ? "top-[75px] opacity-100"
            : "invisible top-[150px] opacity-0"
        )}
      >
        <div className="flex h-full flex-col items-center gap-[10px]">
          {isAuth ? (
            <Link
              to={{ pathname: Routes.signIn }}
              className="w-full outline-none focus:border-0"
            >
              <Button variant="ghost" className="uppercase w-full">
                Личный кабинет
              </Button>
            </Link>
          ) : (
            <>
              <Link
                to={{ pathname: Routes.recognition }}
                className="w-full outline-none focus:border-0"
              >
                <Button variant="ghost" className="uppercase w-full">
                  Демонстрация
                </Button>
              </Link>

              <Link
                to={{ pathname: Routes.signUp }}
                className="w-full outline-none focus:border-0"
              >
                <Button variant="ghost" className="uppercase w-full">
                  Регистрация
                </Button>
              </Link>

              <Link
                to={{ pathname: Routes.signIn }}
                className="w-full outline-none focus:border-0"
              >
                <Button className="uppercase w-full">Авторизация</Button>
              </Link>
            </>
          )}

          {isMainPage &&
            mainPageSections.map(({ sectionId, sectionName }) => (
              <a
                key={sectionId}
                href={`#${sectionId}`}
                className="w-full outline-none focus:border-0"
              >
                <Button variant="ghost" className="uppercase w-full">
                  {sectionName}
                </Button>
              </a>
            ))}
        </div>
      </nav>
    </Portal.Root>
  );
};
