export type SignUpFormFields =
  | "firstName"
  | "lastName"
  | "middleName"
  | "email"
  | "password"
  | "repeatPassword"
  | "agreementTerms";
type SignUpFormFieldsData = Record<
  SignUpFormFields,
  {
    name: SignUpFormFields;
    label: string;
  }
>;

export const signUpFields: SignUpFormFieldsData = {
  firstName: {
    name: "firstName",
    label: "Имя",
  },
  lastName: {
    name: "lastName",
    label: "Фамилия",
  },
  middleName: {
    name: "middleName",
    label: "Отчество",
  },
  email: {
    name: "email",
    label: "E-mail",
  },
  password: {
    name: "password",
    label: "Пароль",
  },
  repeatPassword: {
    name: "repeatPassword",
    label: "Повторите пароль",
  },
  agreementTerms: {
    name: "agreementTerms",
    label: "",
  },
};

export const termsLink = "";
