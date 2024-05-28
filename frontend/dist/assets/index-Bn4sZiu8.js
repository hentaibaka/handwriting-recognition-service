import{j as s,B as n,c as r,g as b,a as h,R as N,D as w,z as f,A as v,E as F,F as I,G as D,I as P}from"./index-DyUJ1aww.js";import{F as p,M as x,a as c,u as y}from"./validation-Dn8NcgNc.js";import{u as E}from"./useMutation-INpl9ai5.js";import"./utils-km2FGkQ4.js";const S=e=>s.jsxs(n,{type:"button",className:r("bg-additional-color-1 text-accent hover:bg-additional-color-1","disabled:bg-additional-color-1/80 disabled:text-accent/60"),...e,children:[s.jsx(b,{iconName:"vk-logo",className:r("mr-[10px] h-[30px] w-[30px]",e.disabled&&"opacity-60")}),"Войти через VK ID"]}),m={email:{name:"email",label:"E-mail"},password:{name:"password",label:"Пароль"}},T=({control:e,disabled:a,onFieldChange:i})=>s.jsx(p,{control:e,name:m.email.name,label:m.email.label,rules:{required:"Это поле обязательное",minLength:{value:x,message:`Поле не может быть короче ${x} символов`}},disabled:a,onFieldChange:i}),L=({control:e,disabled:a,onFieldChange:i})=>s.jsx(p,{control:e,name:m.password.name,label:m.password.label,rules:{required:"Это поле обязательное",minLength:{value:c,message:`Пароль не может быть короче ${c} символов`}},disabled:a,onFieldChange:i}),R=()=>{const e=h(),{isPending:a,mutate:i}=E({mutationFn:t=>new Promise(l=>setTimeout(()=>l(t),3e3)),onSuccess:()=>{e({pathname:N.recognition})}});return{isPending:a,signIn:i}},_={email:"",password:""},M=({className:e,renderButtonSlot:a})=>{const{control:i,handleSubmit:t}=y({defaultValues:_}),{isPending:l,signIn:g}=R(),u=d=>{const j={email:d.email,password:d.password};g(j)};return s.jsxs("form",{className:e,onSubmit:t(u),children:[s.jsxs("div",{className:r("mb-[26px] grid gap-x-[17px] gap-y-[12px] sm:gap-y-[20px] md:gap-y-[30px]","sm:grid-cols-2 xl:grid-cols-[repeat(2,_403px)]"),children:[s.jsx(T,{disabled:l,control:i}),s.jsx(L,{disabled:l,control:i})]}),s.jsxs("div",{className:"mb-[26px] flex flex-wrap gap-[23px]",children:[s.jsx(n,{className:"uppercase",disabled:l,children:"Авторизоваться"}),a==null?void 0:a({disabled:l})]})]})},o="rememberPassword",C=()=>s.jsxs("div",{className:"py-[30px] px-[16px] sm:py-[30px] sm:px-[50px] lg:py-[27px] lg:px-[66px]",children:[s.jsx(V,{}),s.jsx(M,{renderButtonSlot:({disabled:e})=>s.jsx(S,{disabled:e})}),s.jsx(q,{})]}),V=()=>s.jsx("div",{className:r("mb-[13px] text-additional-color-2 font-medium","text-[1.1rem] md:text-[1.3rem] lg:text-[1.375rem] lg:leading-[1.75rem]"),children:"Авторизация"}),q=()=>s.jsxs(w,{children:[s.jsx(f,{asChild:!0,children:s.jsx(n,{variant:"link",className:"!p-0 !py-2 !h-fit !text-[1rem]",children:"Забыли пароль?"})}),s.jsxs(v,{className:"max-w-[708px]",children:[s.jsxs(F,{children:[s.jsx(I,{children:"Забыли пароль?"}),s.jsx(D,{children:"Мы отправим вам новый пароль на вашу почту"})]}),s.jsxs("div",{className:"mb-[20px] flex flex-wrap sm:flex-nowrap items-end gap-[13px]",children:[s.jsxs("div",{className:"w-full",children:[s.jsx("label",{htmlFor:o,className:"mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans",children:"E-mail"}),s.jsx(P,{id:o,name:o})]}),s.jsx(n,{className:"uppercase",children:"Отправить"})]})]})]});export{C as default};