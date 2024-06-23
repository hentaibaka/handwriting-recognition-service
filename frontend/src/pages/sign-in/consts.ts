export type SignInFormFields = "email" | "password";
type SignInFormFieldsData = Record<
  SignInFormFields,
  {
    name: SignInFormFields;
    label: string;
  }
>;

export const signInFields: SignInFormFieldsData = {
  email: {
    name: "email",
    label: "E-mail",
  },
  password: {
    name: "password",
    label: "Пароль",
  },
};
