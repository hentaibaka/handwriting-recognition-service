import{a as D,O as T,f as $,g as V,P as v,u as L,h as E,R as b,j as e,c as f,i as k,L as K,r as y,I as U,B as S,k as G,S as H}from"./index-DUDia-6_.js";import{F as p,M as m,a as w,u as Q,d as W,s as z}from"./validation-Dc6vHycW.js";import{P as J,a as j}from"./PasswordComplexity-Cohp1xyn.js";import{u as F}from"./useMutation-BPfTpUlQ.js";class X{static logoutCreate(){return D(T,{method:"POST",url:"/api/logout/"})}}const Y=()=>{const s=$(),{toast:t}=V(),{isPending:n,mutateAsync:a}=F({mutationKey:["profile-update"],mutationFn:v.profileUpdateUserUpdate,onError:o=>{const l=o;l.body||console.error(o);for(const g in l.body)t({title:"Ошибка обновления профиля",description:l.body[g]||"Неизвестная ошибка"})}}),{isPending:d,mutateAsync:u}=F({mutationFn:v.profileChangePasswordCreate,onError:o=>{const l=o;l.body||console.error(o);for(const g in l.body)t({title:"Ошибка обновления профиля",description:l.body[g]||"Неизвестная ошибка"})}});return{isProfileDataPending:n||d,updateProfileData:async o=>{await a(o),o.new_password&&o.old_password&&await u(o),s.invalidateQueries({queryKey:["viewer"]})}}},Z=()=>{const s=L(),{setIsAuth:t}=E(),{mutate:n,isPending:a}=F({mutationFn:X.logoutCreate,onSuccess(){t(!1),s({pathname:b.signIn})}});return{signOut:n,isSignOutPending:a}},ee=({className:s,currentPath:t,links:n})=>e.jsx("div",{className:f("font-deja-vu-sans font-extralight text-additional-color-2 text-[0.8rem] sm:text-[0.95rem] leading-[1.1rem]","flex gap-[12px]",s),children:n.map(({label:a,path:d},u)=>e.jsxs(k.Fragment,{children:[e.jsx(K,{to:{pathname:d},className:f(t===d&&"text-accent font-normal"),children:a}),n.length-1!==u&&e.jsx("span",{className:"font-inter font-light text-[0.8rem] leading-[0.96rem]",children:">"})]},d))}),i={status:{name:"status",label:"Статус"},email:{name:"email",label:"E-mail"},lastName:{name:"lastName",label:"Фамилия"},firstName:{name:"firstName",label:"Имя"},middleName:{name:"middleName",label:"Отчество"},currentPassword:{name:"currentPassword",label:"Текущий пароль"},newPassword:{name:"newPassword",label:"Новый пароль"}},se=({className:s,control:t,disabled:n,onFieldChange:a})=>e.jsx(p,{className:s,control:t,name:i.firstName.name,label:i.firstName.label,rules:{required:"Это поле обязательное",minLength:{value:m,message:`Поле не может быть короче ${m} символов`}},disabled:n,onFieldChange:a}),ae=({className:s,control:t,disabled:n,onFieldChange:a})=>e.jsx(p,{className:s,control:t,name:i.lastName.name,label:i.lastName.label,rules:{required:"Это поле обязательное",minLength:{value:m,message:`Поле не может быть короче ${m} символов`}},disabled:n,onFieldChange:a}),te=({className:s,control:t,disabled:n,onFieldChange:a})=>e.jsx(p,{className:s,control:t,name:i.middleName.name,label:i.middleName.label,rules:{required:"Это поле обязательное",minLength:{value:m,message:`Поле не может быть короче ${m} символов`}},disabled:n,onFieldChange:a}),ne=({className:s,control:t,disabled:n,onFieldChange:a})=>e.jsx(p,{className:s,control:t,name:i.email.name,label:i.email.label,rules:{required:"Это поле обязательное",minLength:{value:m,message:`Поле не может быть короче ${m} символов`}},disabled:n,onFieldChange:a}),le=({className:s,control:t,disabled:n,onFieldChange:a})=>e.jsx(p,{className:s,control:t,name:i.currentPassword.name,label:i.currentPassword.label,rules:{minLength:{value:w,message:`Пароль не может быть короче ${w} символов`}},disabled:n,onFieldChange:a}),re=({className:s,control:t,disabled:n,onFieldChange:a})=>e.jsx(p,{className:s,control:t,name:i.newPassword.name,label:i.newPassword.label,rules:{minLength:{value:w,message:`Пароль не может быть короче ${w} символов`}},disabled:n,onFieldChange:a}),ie=({viewerData:s,signOutButton:t})=>{const n={email:s.email||"",firstName:s.firstName||"",lastName:s.lastName||"",middleName:s.middleName||"",currentPassword:"",newPassword:"",status:""},{control:a,handleSubmit:d,getValues:u}=Q({defaultValues:n,reValidateMode:"onChange",mode:"onChange"}),{isProfileDataPending:c,updateProfileData:o}=Y(),[l,g]=y.useState({minPasswordLength:!1,notIncludeNameAndEmail:!0,containsNumberOrSymbol:!1,passwordStrength:"Слабая"}),[M,O]=y.useState(!1),I=r=>{const x=r.length>=w,h=u(i.email.name),N=u(i.firstName.name),P=!!h&&r.includes(h),R=!!N&&r.includes(N),A=!P&&!R,_=W.test(r),B=z.test(r);return{isMatchPasswordLength:x,isMatchNotIncludeNameAndEmail:A,isMatchContainsNumberOrSymbol:_||B}},C=r=>{O(!0);const{isMatchContainsNumberOrSymbol:x,isMatchNotIncludeNameAndEmail:h,isMatchPasswordLength:N}=I(r),P=[N,h,x].filter(Boolean).length;g({containsNumberOrSymbol:x,notIncludeNameAndEmail:h,minPasswordLength:N,passwordStrength:P<=1?"Слабая":P===2?"Нормальная":"Сильная"})},q=r=>{const x={email:r.email,first_name:r.firstName,last_name:r.lastName,middle_name:r.middleName,new_password:r.newPassword,old_password:r.currentPassword};o(x)};return e.jsxs("form",{onSubmit:d(q),children:[e.jsxs("div",{className:"mb-[38px] profile-page-fields-grid gap-x-[17px] gap-y-[12px] sm:gap-y-[20px] md:gap-y-[30px]",children:[e.jsxs("div",{className:"xl:w-[318px] status",children:[e.jsx("label",{className:"mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans",children:"Статус"}),e.jsx(U,{className:"!cursor-default border-accent",disabled:!0,value:s.status})]}),e.jsx(ne,{className:"xl:w-[318px] email",control:a,disabled:c}),e.jsx(ae,{className:"xl:w-[318px] lastName",control:a,disabled:c}),e.jsx(se,{className:"xl:w-[318px] firstName",control:a,disabled:c}),e.jsx(te,{className:"xl:w-[318px] middleName",control:a,disabled:c}),e.jsx(le,{className:"xl:w-[318px] password",control:a,disabled:c,onFieldChange:C}),e.jsx(re,{className:"xl:w-[318px] newPassword",control:a,disabled:c,onFieldChange:C}),e.jsxs(J,{className:f("password-complexity",!M&&"invisible"),children:[e.jsx(j,{isMeetRequirements:l.passwordStrength!=="Слабая",customLabel:e.jsxs("div",{children:[e.jsx("span",{children:"Надежность пароля: "}),e.jsx("span",{className:f(l.passwordStrength==="Слабая"&&"text-error"),children:l.passwordStrength})]})}),e.jsx(j,{isMeetRequirements:l.minPasswordLength,label:"Минимум 8 символов"}),e.jsx(j,{isMeetRequirements:l.notIncludeNameAndEmail,label:"Не может содержать ваше имя или адрес электронной почты"}),e.jsx(j,{isMeetRequirements:l.containsNumberOrSymbol,label:"Содержит число или символ"})]})]}),e.jsxs("div",{className:"flex flex-wrap flex-col sm:flex-row gap-[12px] sm:gap-[20px]",children:[e.jsx(S,{type:"button",children:"История распознавания"}),e.jsx(S,{type:"submit",children:"Редактировать"}),t]})]})},oe=()=>{const{isSignOutPending:s,signOut:t}=Z();return e.jsx(S,{type:"button",disabled:s,onClick:()=>t(),children:"Выйти"})},Ne=()=>{const{viewerData:s,isLoadingViewer:t}=G(),{isAuth:n}=E(),a=L();return y.useEffect(()=>{n||a({pathname:b.signIn})},[n,a]),e.jsxs("div",{className:"mb-[40px]",children:[e.jsx(me,{}),e.jsx(ee,{className:"sm:mb-[29px]",currentPath:b.profile,links:ce}),e.jsxs("div",{className:"py-[30px] px-[16px] sm:py-[30px] sm:px-[50px] lg:py-[27px] lg:px-[66px]",children:[e.jsx(de,{}),t?e.jsx("div",{className:"h-[300px] grid place-content-center",children:e.jsx(H,{className:"!w-16 !h-16 border-4"})}):e.jsx(ie,{viewerData:s,signOutButton:e.jsx(oe,{})})]})]})},me=()=>e.jsx("h1",{className:"mb-[16px] font-medium text-[1.2rem] sm:text-[1.875rem] leading-[2.5rem]",children:"Личный кабинет"}),de=()=>e.jsx("div",{className:f("mb-[22px] text-additional-color-2 font-medium","text-[1.1rem] md:text-[1.3rem] lg:text-[1.375rem] lg:leading-[1.75rem]"),children:"Личные данные"}),ce=[{label:"Главная",path:b.main},{label:"Личный кабинет",path:b.profile}];export{Ne as default};