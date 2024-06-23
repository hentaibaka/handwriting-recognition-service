export type ProfileFormFields =
  | "status"
  | "email"
  | "lastName"
  | "firstName"
  | "middleName"
  | "currentPassword"
  | "newPassword";
type ProfileFormFieldsData = Record<
  ProfileFormFields,
  {
    name: ProfileFormFields;
    label: string;
  }
>;

export const profileFormFields: ProfileFormFieldsData = {
  status: {
    name: "status",
    label: "Статус",
  },
  email: {
    name: "email",
    label: "E-mail",
  },
  lastName: {
    name: "lastName",
    label: "Фамилия",
  },
  firstName: {
    name: "firstName",
    label: "Имя",
  },
  middleName: {
    name: "middleName",
    label: "Отчество",
  },
  currentPassword: {
    name: "currentPassword",
    label: "Текущий пароль",
  },
  newPassword: {
    name: "newPassword",
    label: "Новый пароль",
  },
};
