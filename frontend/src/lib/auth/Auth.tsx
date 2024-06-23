import { useViewer } from "@/hooks";
import { createContext, ReactNode, useEffect, useState } from "react";

interface AuthContextType {
  isAuth: boolean;
  setIsAuth: (value: boolean) => void;
}

const localStorageAuthKey = "isAuth";

export const AuthContext = createContext<AuthContextType>({
  isAuth: false,
  setIsAuth: () => {},
});

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [isAuth, updateIsAuth] = useState(false);

  const setIsAuth = (value: boolean) => {
    if (value) {
      updateIsAuth(true);
      localStorage.setItem(localStorageAuthKey, "true");
      return;
    }

    updateIsAuth(false);
    localStorage.removeItem(localStorageAuthKey);
  };

  useEffect(() => {
    const isAuthFromLS = localStorage.getItem(localStorageAuthKey) === "true";

    if (isAuthFromLS) {
      return updateIsAuth(true);
    }

    localStorage.removeItem(localStorageAuthKey);
  }, []);

  useViewer();

  return (
    <AuthContext.Provider value={{ isAuth, setIsAuth }}>
      {children}
    </AuthContext.Provider>
  );
};
