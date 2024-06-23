import { Outlet } from "react-router-dom";
import { Spinner } from "@/components";
import { Container, Footer, Header } from "./components";
import { Suspense } from "react";

export const MainLayout = () => {
  return (
    <>
      <Header />

      <Container className="flex flex-col flex-grow">
        <Suspense fallback={<Fallback />}>
          <Outlet />
        </Suspense>
      </Container>

      <Footer />
    </>
  );
};

const Fallback = () => (
  <div className="w-full flex-grow flex">
    <div className="flex flex-grow justify-center items-center">
      <Spinner className="!h-20 !w-20 border-4" />
    </div>
  </div>
);
