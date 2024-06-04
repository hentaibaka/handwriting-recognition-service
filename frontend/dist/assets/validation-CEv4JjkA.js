import{f as D,j as ae,F as mt,c as At}from"./index-DZk8EHuN.js";var ce=e=>e.type==="checkbox",ie=e=>e instanceof Date,O=e=>e==null;const tt=e=>typeof e=="object";var R=e=>!O(e)&&!Array.isArray(e)&&tt(e)&&!ie(e),rt=e=>R(e)&&e.target?ce(e.target)?e.target.checked:e.target.value:e,Dt=e=>e.substring(0,e.search(/\.\d+(\.|$)/))||e,st=(e,s)=>e.has(Dt(s)),wt=e=>{const s=e.constructor&&e.constructor.prototype;return R(s)&&s.hasOwnProperty("isPrototypeOf")},Le=typeof window<"u"&&typeof window.HTMLElement<"u"&&typeof document<"u";function U(e){let s;const t=Array.isArray(e);if(e instanceof Date)s=new Date(e);else if(e instanceof Set)s=new Set(e);else if(!(Le&&(e instanceof Blob||e instanceof FileList))&&(t||R(e)))if(s=t?[]:{},!t&&!wt(e))s=e;else for(const a in e)e.hasOwnProperty(a)&&(s[a]=U(e[a]));else return e;return s}var de=e=>Array.isArray(e)?e.filter(Boolean):[],S=e=>e===void 0,f=(e,s,t)=>{if(!s||!R(e))return t;const a=de(s.split(/[,[\].]+?/)).reduce((u,l)=>O(u)?u:u[l],e);return S(a)||a===e?S(e[s])?t:e[s]:a},q=e=>typeof e=="boolean";const be={BLUR:"blur",FOCUS_OUT:"focusout",CHANGE:"change"},W={onBlur:"onBlur",onChange:"onChange",onSubmit:"onSubmit",onTouched:"onTouched",all:"all"},z={max:"max",min:"min",maxLength:"maxLength",minLength:"minLength",pattern:"pattern",required:"required",validate:"validate"},St=D.createContext(null),Te=()=>D.useContext(St);var it=(e,s,t,a=!0)=>{const u={defaultValues:s._defaultValues};for(const l in e)Object.defineProperty(u,l,{get:()=>{const y=l;return s._proxyFormState[y]!==W.all&&(s._proxyFormState[y]=!a||W.all),t&&(t[y]=!0),e[y]}});return u},N=e=>R(e)&&!Object.keys(e).length,at=(e,s,t,a)=>{t(e);const{name:u,...l}=e;return N(l)||Object.keys(l).length>=Object.keys(s).length||Object.keys(l).find(y=>s[y]===(!a||W.all))},ve=e=>Array.isArray(e)?e:[e],ut=(e,s,t)=>!e||!s||e===s||ve(e).some(a=>a&&(t?a===s:a.startsWith(s)||s.startsWith(a)));function pe(e){const s=D.useRef(e);s.current=e,D.useEffect(()=>{const t=!e.disabled&&s.current.subject&&s.current.subject.subscribe({next:s.current.next});return()=>{t&&t.unsubscribe()}},[e.disabled])}function Et(e){const s=Te(),{control:t=s.control,disabled:a,name:u,exact:l}=e||{},[y,g]=D.useState(t._formState),V=D.useRef(!0),w=D.useRef({isDirty:!1,isLoading:!1,dirtyFields:!1,touchedFields:!1,validatingFields:!1,isValidating:!1,isValid:!1,errors:!1}),_=D.useRef(u);return _.current=u,pe({disabled:a,next:v=>V.current&&ut(_.current,v.name,l)&&at(v,w.current,t._updateFormState)&&g({...t._formState,...v}),subject:t._subjects.state}),D.useEffect(()=>(V.current=!0,w.current.isValid&&t._updateValid(!0),()=>{V.current=!1}),[t]),it(y,t,w.current,!1)}var $=e=>typeof e=="string",lt=(e,s,t,a,u)=>$(e)?(a&&s.watch.add(e),f(t,e,u)):Array.isArray(e)?e.map(l=>(a&&s.watch.add(l),f(t,l))):(a&&(s.watchAll=!0),t);function kt(e){const s=Te(),{control:t=s.control,name:a,defaultValue:u,disabled:l,exact:y}=e||{},g=D.useRef(a);g.current=a,pe({disabled:l,subject:t._subjects.values,next:_=>{ut(g.current,_.name,y)&&w(U(lt(g.current,t._names,_.values||t._formValues,!1,u)))}});const[V,w]=D.useState(t._getWatch(a,u));return D.useEffect(()=>t._removeUnmounted()),V}var Oe=e=>/^\w*$/.test(e),nt=e=>de(e.replace(/["|']|\]/g,"").split(/\.|\[/)),A=(e,s,t)=>{let a=-1;const u=Oe(s)?[s]:nt(s),l=u.length,y=l-1;for(;++a<l;){const g=u[a];let V=t;if(a!==y){const w=e[g];V=R(w)||Array.isArray(w)?w:isNaN(+u[a+1])?{}:[]}e[g]=V,e=e[g]}return e};function Ct(e){const s=Te(),{name:t,disabled:a,control:u=s.control,shouldUnregister:l}=e,y=st(u._names.array,t),g=kt({control:u,name:t,defaultValue:f(u._formValues,t,f(u._defaultValues,t,e.defaultValue)),exact:!0}),V=Et({control:u,name:t}),w=D.useRef(u.register(t,{...e.rules,value:g,...q(e.disabled)?{disabled:e.disabled}:{}}));return D.useEffect(()=>{const _=u._options.shouldUnregister||l,v=(P,H)=>{const T=f(u._fields,P);T&&(T._f.mount=H)};if(v(t,!0),_){const P=U(f(u._options.defaultValues,t));A(u._defaultValues,t,P),S(f(u._formValues,t))&&A(u._formValues,t,P)}return()=>{(y?_&&!u._state.action:_)?u.unregister(t):v(t,!1)}},[t,u,y,l]),D.useEffect(()=>{f(u._fields,t)&&u._updateDisabledField({disabled:a,fields:u._fields,name:t,value:f(u._fields,t)._f.value})},[a,t,u]),{field:{name:t,value:g,...q(a)||V.disabled?{disabled:V.disabled||a}:{},onChange:D.useCallback(_=>w.current.onChange({target:{value:rt(_),name:t},type:be.CHANGE}),[t]),onBlur:D.useCallback(()=>w.current.onBlur({target:{value:f(u._formValues,t),name:t},type:be.BLUR}),[t,u]),ref:_=>{const v=f(u._fields,t);v&&_&&(v._f.ref={focus:()=>_.focus(),select:()=>_.select(),setCustomValidity:P=>_.setCustomValidity(P),reportValidity:()=>_.reportValidity()})}},formState:V,fieldState:Object.defineProperties({},{invalid:{enumerable:!0,get:()=>!!f(V.errors,t)},isDirty:{enumerable:!0,get:()=>!!f(V.dirtyFields,t)},isTouched:{enumerable:!0,get:()=>!!f(V.touchedFields,t)},isValidating:{enumerable:!0,get:()=>!!f(V.validatingFields,t)},error:{enumerable:!0,get:()=>f(V.errors,t)}})}}const Rt=e=>e.render(Ct(e));var Lt=(e,s,t,a,u)=>s?{...t[e],types:{...t[e]&&t[e].types?t[e].types:{},[a]:u||!0}}:{},Ke=e=>({isOnSubmit:!e||e===W.onSubmit,isOnBlur:e===W.onBlur,isOnChange:e===W.onChange,isOnAll:e===W.all,isOnTouch:e===W.onTouched}),ze=(e,s,t)=>!t&&(s.watchAll||s.watch.has(e)||[...s.watch].some(a=>e.startsWith(a)&&/^\.\w+/.test(e.slice(a.length))));const fe=(e,s,t,a)=>{for(const u of t||Object.keys(e)){const l=f(e,u);if(l){const{_f:y,...g}=l;if(y){if(y.refs&&y.refs[0]&&s(y.refs[0],u)&&!a)break;if(y.ref&&s(y.ref,y.name)&&!a)break;fe(g,s)}else R(g)&&fe(g,s)}}};var Tt=(e,s,t)=>{const a=de(f(e,t));return A(a,"root",s[t]),A(e,t,a),e},Ue=e=>e.type==="file",X=e=>typeof e=="function",Ve=e=>{if(!Le)return!1;const s=e?e.ownerDocument:0;return e instanceof(s&&s.defaultView?s.defaultView.HTMLElement:HTMLElement)},_e=e=>$(e),Me=e=>e.type==="radio",Fe=e=>e instanceof RegExp;const Je={value:!1,isValid:!1},Qe={value:!0,isValid:!0};var ot=e=>{if(Array.isArray(e)){if(e.length>1){const s=e.filter(t=>t&&t.checked&&!t.disabled).map(t=>t.value);return{value:s,isValid:!!s.length}}return e[0].checked&&!e[0].disabled?e[0].attributes&&!S(e[0].attributes.value)?S(e[0].value)||e[0].value===""?Qe:{value:e[0].value,isValid:!0}:Qe:Je}return Je};const Xe={isValid:!1,value:null};var ft=e=>Array.isArray(e)?e.reduce((s,t)=>t&&t.checked&&!t.disabled?{isValid:!0,value:t.value}:s,Xe):Xe;function Ye(e,s,t="validate"){if(_e(e)||Array.isArray(e)&&e.every(_e)||q(e)&&!e)return{type:t,message:_e(e)?e:"",ref:s}}var se=e=>R(e)&&!Fe(e)?e:{value:e,message:""},Ze=async(e,s,t,a,u)=>{const{ref:l,refs:y,required:g,maxLength:V,minLength:w,min:_,max:v,pattern:P,validate:H,name:T,valueAsNumber:Ae,mount:J,disabled:Q}=e._f,F=f(s,T);if(!J||Q)return{};const G=y?y[0]:l,K=b=>{a&&G.reportValidity&&(G.setCustomValidity(q(b)?"":b||""),G.reportValidity())},E={},ee=Me(l),ye=ce(l),Y=ee||ye,te=(Ae||Ue(l))&&S(l.value)&&S(F)||Ve(l)&&l.value===""||F===""||Array.isArray(F)&&!F.length,B=Lt.bind(null,T,t,E),ge=(b,x,k,p=z.maxLength,j=z.minLength)=>{const I=b?x:k;E[T]={type:b?p:j,message:I,ref:l,...B(b?p:j,I)}};if(u?!Array.isArray(F)||!F.length:g&&(!Y&&(te||O(F))||q(F)&&!F||ye&&!ot(y).isValid||ee&&!ft(y).isValid)){const{value:b,message:x}=_e(g)?{value:!!g,message:g}:se(g);if(b&&(E[T]={type:z.required,message:x,ref:G,...B(z.required,x)},!t))return K(x),E}if(!te&&(!O(_)||!O(v))){let b,x;const k=se(v),p=se(_);if(!O(F)&&!isNaN(F)){const j=l.valueAsNumber||F&&+F;O(k.value)||(b=j>k.value),O(p.value)||(x=j<p.value)}else{const j=l.valueAsDate||new Date(F),I=ne=>new Date(new Date().toDateString()+" "+ne),ue=l.type=="time",le=l.type=="week";$(k.value)&&F&&(b=ue?I(F)>I(k.value):le?F>k.value:j>new Date(k.value)),$(p.value)&&F&&(x=ue?I(F)<I(p.value):le?F<p.value:j<new Date(p.value))}if((b||x)&&(ge(!!b,k.message,p.message,z.max,z.min),!t))return K(E[T].message),E}if((V||w)&&!te&&($(F)||u&&Array.isArray(F))){const b=se(V),x=se(w),k=!O(b.value)&&F.length>+b.value,p=!O(x.value)&&F.length<+x.value;if((k||p)&&(ge(k,b.message,x.message),!t))return K(E[T].message),E}if(P&&!te&&$(F)){const{value:b,message:x}=se(P);if(Fe(b)&&!F.match(b)&&(E[T]={type:z.pattern,message:x,ref:l,...B(z.pattern,x)},!t))return K(x),E}if(H){if(X(H)){const b=await H(F,s),x=Ye(b,G);if(x&&(E[T]={...x,...B(z.validate,x.message)},!t))return K(x.message),E}else if(R(H)){let b={};for(const x in H){if(!N(b)&&!t)break;const k=Ye(await H[x](F,s),G,x);k&&(b={...k,...B(x,k.message)},K(k.message),t&&(E[T]=b))}if(!N(b)&&(E[T]={ref:G,...b},!t))return E}}return K(!0),E};function pt(e,s){const t=s.slice(0,-1).length;let a=0;for(;a<t;)e=S(e)?a++:e[s[a++]];return e}function Ot(e){for(const s in e)if(e.hasOwnProperty(s)&&!S(e[s]))return!1;return!0}function C(e,s){const t=Array.isArray(s)?s:Oe(s)?[s]:nt(s),a=t.length===1?e:pt(e,t),u=t.length-1,l=t[u];return a&&delete a[l],u!==0&&(R(a)&&N(a)||Array.isArray(a)&&Ot(a))&&C(e,t.slice(0,-1)),e}var ke=()=>{let e=[];return{get observers(){return e},next:u=>{for(const l of e)l.next&&l.next(u)},subscribe:u=>(e.push(u),{unsubscribe:()=>{e=e.filter(l=>l!==u)}}),unsubscribe:()=>{e=[]}}},xe=e=>O(e)||!tt(e);function Z(e,s){if(xe(e)||xe(s))return e===s;if(ie(e)&&ie(s))return e.getTime()===s.getTime();const t=Object.keys(e),a=Object.keys(s);if(t.length!==a.length)return!1;for(const u of t){const l=e[u];if(!a.includes(u))return!1;if(u!=="ref"){const y=s[u];if(ie(l)&&ie(y)||R(l)&&R(y)||Array.isArray(l)&&Array.isArray(y)?!Z(l,y):l!==y)return!1}}return!0}var ct=e=>e.type==="select-multiple",Ut=e=>Me(e)||ce(e),Ce=e=>Ve(e)&&e.isConnected,dt=e=>{for(const s in e)if(X(e[s]))return!0;return!1};function me(e,s={}){const t=Array.isArray(e);if(R(e)||t)for(const a in e)Array.isArray(e[a])||R(e[a])&&!dt(e[a])?(s[a]=Array.isArray(e[a])?[]:{},me(e[a],s[a])):O(e[a])||(s[a]=!0);return s}function yt(e,s,t){const a=Array.isArray(e);if(R(e)||a)for(const u in e)Array.isArray(e[u])||R(e[u])&&!dt(e[u])?S(s)||xe(t[u])?t[u]=Array.isArray(e[u])?me(e[u],[]):{...me(e[u])}:yt(e[u],O(s)?{}:s[u],t[u]):t[u]=!Z(e[u],s[u]);return t}var he=(e,s)=>yt(e,s,me(s)),gt=(e,{valueAsNumber:s,valueAsDate:t,setValueAs:a})=>S(e)?e:s?e===""?NaN:e&&+e:t&&$(e)?new Date(e):a?a(e):e;function Re(e){const s=e.ref;if(!(e.refs?e.refs.every(t=>t.disabled):s.disabled))return Ue(s)?s.files:Me(s)?ft(e.refs).value:ct(s)?[...s.selectedOptions].map(({value:t})=>t):ce(s)?ot(e.refs).value:gt(S(s.value)?e.ref.value:s.value,e)}var Mt=(e,s,t,a)=>{const u={};for(const l of e){const y=f(s,l);y&&A(u,l,y._f)}return{criteriaMode:t,names:[...e],fields:u,shouldUseNativeValidation:a}},oe=e=>S(e)?e:Fe(e)?e.source:R(e)?Fe(e.value)?e.value.source:e.value:e,Nt=e=>e.mount&&(e.required||e.min||e.max||e.maxLength||e.minLength||e.pattern||e.validate);function et(e,s,t){const a=f(e,t);if(a||Oe(t))return{error:a,name:t};const u=t.split(".");for(;u.length;){const l=u.join("."),y=f(s,l),g=f(e,l);if(y&&!Array.isArray(y)&&t!==l)return{name:t};if(g&&g.type)return{name:l,error:g};u.pop()}return{name:t}}var Bt=(e,s,t,a,u)=>u.isOnAll?!1:!t&&u.isOnTouch?!(s||e):(t?a.isOnBlur:u.isOnBlur)?!e:(t?a.isOnChange:u.isOnChange)?e:!0,It=(e,s)=>!de(f(e,s)).length&&C(e,s);const Pt={mode:W.onSubmit,reValidateMode:W.onChange,shouldFocusError:!0};function jt(e={}){let s={...Pt,...e},t={submitCount:0,isDirty:!1,isLoading:X(s.defaultValues),isValidating:!1,isSubmitted:!1,isSubmitting:!1,isSubmitSuccessful:!1,isValid:!1,touchedFields:{},dirtyFields:{},validatingFields:{},errors:s.errors||{},disabled:s.disabled||!1},a={},u=R(s.defaultValues)||R(s.values)?U(s.defaultValues||s.values)||{}:{},l=s.shouldUnregister?{}:U(u),y={action:!1,mount:!1,watch:!1},g={mount:new Set,unMount:new Set,array:new Set,watch:new Set},V,w=0;const _={isDirty:!1,dirtyFields:!1,validatingFields:!1,touchedFields:!1,isValidating:!1,isValid:!1,errors:!1},v={values:ke(),array:ke(),state:ke()},P=Ke(s.mode),H=Ke(s.reValidateMode),T=s.criteriaMode===W.all,Ae=r=>i=>{clearTimeout(w),w=setTimeout(r,i)},J=async r=>{if(_.isValid||r){const i=s.resolver?N((await Y()).errors):await B(a,!0);i!==t.isValid&&v.state.next({isValid:i})}},Q=(r,i)=>{(_.isValidating||_.validatingFields)&&((r||Array.from(g.mount)).forEach(n=>{n&&(i?A(t.validatingFields,n,i):C(t.validatingFields,n))}),v.state.next({validatingFields:t.validatingFields,isValidating:!N(t.validatingFields)}))},F=(r,i=[],n,d,c=!0,o=!0)=>{if(d&&n){if(y.action=!0,o&&Array.isArray(f(a,r))){const h=n(f(a,r),d.argA,d.argB);c&&A(a,r,h)}if(o&&Array.isArray(f(t.errors,r))){const h=n(f(t.errors,r),d.argA,d.argB);c&&A(t.errors,r,h),It(t.errors,r)}if(_.touchedFields&&o&&Array.isArray(f(t.touchedFields,r))){const h=n(f(t.touchedFields,r),d.argA,d.argB);c&&A(t.touchedFields,r,h)}_.dirtyFields&&(t.dirtyFields=he(u,l)),v.state.next({name:r,isDirty:b(r,i),dirtyFields:t.dirtyFields,errors:t.errors,isValid:t.isValid})}else A(l,r,i)},G=(r,i)=>{A(t.errors,r,i),v.state.next({errors:t.errors})},K=r=>{t.errors=r,v.state.next({errors:t.errors,isValid:!1})},E=(r,i,n,d)=>{const c=f(a,r);if(c){const o=f(l,r,S(n)?f(u,r):n);S(o)||d&&d.defaultChecked||i?A(l,r,i?o:Re(c._f)):p(r,o),y.mount&&J()}},ee=(r,i,n,d,c)=>{let o=!1,h=!1;const m={name:r},L=!!(f(a,r)&&f(a,r)._f.disabled);if(!n||d){_.isDirty&&(h=t.isDirty,t.isDirty=m.isDirty=b(),o=h!==m.isDirty);const M=L||Z(f(u,r),i);h=!!(!L&&f(t.dirtyFields,r)),M||L?C(t.dirtyFields,r):A(t.dirtyFields,r,!0),m.dirtyFields=t.dirtyFields,o=o||_.dirtyFields&&h!==!M}if(n){const M=f(t.touchedFields,r);M||(A(t.touchedFields,r,n),m.touchedFields=t.touchedFields,o=o||_.touchedFields&&M!==n)}return o&&c&&v.state.next(m),o?m:{}},ye=(r,i,n,d)=>{const c=f(t.errors,r),o=_.isValid&&q(i)&&t.isValid!==i;if(e.delayError&&n?(V=Ae(()=>G(r,n)),V(e.delayError)):(clearTimeout(w),V=null,n?A(t.errors,r,n):C(t.errors,r)),(n?!Z(c,n):c)||!N(d)||o){const h={...d,...o&&q(i)?{isValid:i}:{},errors:t.errors,name:r};t={...t,...h},v.state.next(h)}},Y=async r=>{Q(r,!0);const i=await s.resolver(l,s.context,Mt(r||g.mount,a,s.criteriaMode,s.shouldUseNativeValidation));return Q(r),i},te=async r=>{const{errors:i}=await Y(r);if(r)for(const n of r){const d=f(i,n);d?A(t.errors,n,d):C(t.errors,n)}else t.errors=i;return i},B=async(r,i,n={valid:!0})=>{for(const d in r){const c=r[d];if(c){const{_f:o,...h}=c;if(o){const m=g.array.has(o.name);Q([d],!0);const L=await Ze(c,l,T,s.shouldUseNativeValidation&&!i,m);if(Q([d]),L[o.name]&&(n.valid=!1,i))break;!i&&(f(L,o.name)?m?Tt(t.errors,L,o.name):A(t.errors,o.name,L[o.name]):C(t.errors,o.name))}h&&await B(h,i,n)}}return n.valid},ge=()=>{for(const r of g.unMount){const i=f(a,r);i&&(i._f.refs?i._f.refs.every(n=>!Ce(n)):!Ce(i._f.ref))&&De(r)}g.unMount=new Set},b=(r,i)=>(r&&i&&A(l,r,i),!Z(Ne(),u)),x=(r,i,n)=>lt(r,g,{...y.mount?l:S(i)?u:$(r)?{[r]:i}:i},n,i),k=r=>de(f(y.mount?l:u,r,e.shouldUnregister?f(u,r,[]):[])),p=(r,i,n={})=>{const d=f(a,r);let c=i;if(d){const o=d._f;o&&(!o.disabled&&A(l,r,gt(i,o)),c=Ve(o.ref)&&O(i)?"":i,ct(o.ref)?[...o.ref.options].forEach(h=>h.selected=c.includes(h.value)):o.refs?ce(o.ref)?o.refs.length>1?o.refs.forEach(h=>(!h.defaultChecked||!h.disabled)&&(h.checked=Array.isArray(c)?!!c.find(m=>m===h.value):c===h.value)):o.refs[0]&&(o.refs[0].checked=!!c):o.refs.forEach(h=>h.checked=h.value===c):Ue(o.ref)?o.ref.value="":(o.ref.value=c,o.ref.type||v.values.next({name:r,values:{...l}})))}(n.shouldDirty||n.shouldTouch)&&ee(r,c,n.shouldTouch,n.shouldDirty,!0),n.shouldValidate&&ne(r)},j=(r,i,n)=>{for(const d in i){const c=i[d],o=`${r}.${d}`,h=f(a,o);(g.array.has(r)||!xe(c)||h&&!h._f)&&!ie(c)?j(o,c,n):p(o,c,n)}},I=(r,i,n={})=>{const d=f(a,r),c=g.array.has(r),o=U(i);A(l,r,o),c?(v.array.next({name:r,values:{...l}}),(_.isDirty||_.dirtyFields)&&n.shouldDirty&&v.state.next({name:r,dirtyFields:he(u,l),isDirty:b(r,o)})):d&&!d._f&&!O(o)?j(r,o,n):p(r,o,n),ze(r,g)&&v.state.next({...t}),v.values.next({name:y.mount?r:void 0,values:{...l}})},ue=async r=>{y.mount=!0;const i=r.target;let n=i.name,d=!0;const c=f(a,n),o=()=>i.type?Re(c._f):rt(r),h=m=>{d=Number.isNaN(m)||m===f(l,n,m)};if(c){let m,L;const M=o(),re=r.type===be.BLUR||r.type===be.FOCUS_OUT,Vt=!Nt(c._f)&&!s.resolver&&!f(t.errors,n)&&!c._f.deps||Bt(re,f(t.touchedFields,n),t.isSubmitted,H,P),Se=ze(n,g,re);A(l,n,M),re?(c._f.onBlur&&c._f.onBlur(r),V&&V(0)):c._f.onChange&&c._f.onChange(r);const Ee=ee(n,M,re,!1),Ft=!N(Ee)||Se;if(!re&&v.values.next({name:n,type:r.type,values:{...l}}),Vt)return _.isValid&&J(),Ft&&v.state.next({name:n,...Se?{}:Ee});if(!re&&Se&&v.state.next({...t}),s.resolver){const{errors:$e}=await Y([n]);if(h(M),d){const xt=et(t.errors,a,n),Ge=et($e,a,xt.name||n);m=Ge.error,n=Ge.name,L=N($e)}}else Q([n],!0),m=(await Ze(c,l,T,s.shouldUseNativeValidation))[n],Q([n]),h(M),d&&(m?L=!1:_.isValid&&(L=await B(a,!0)));d&&(c._f.deps&&ne(c._f.deps),ye(n,L,m,Ee))}},le=(r,i)=>{if(f(t.errors,i)&&r.focus)return r.focus(),1},ne=async(r,i={})=>{let n,d;const c=ve(r);if(s.resolver){const o=await te(S(r)?r:c);n=N(o),d=r?!c.some(h=>f(o,h)):n}else r?(d=(await Promise.all(c.map(async o=>{const h=f(a,o);return await B(h&&h._f?{[o]:h}:h)}))).every(Boolean),!(!d&&!t.isValid)&&J()):d=n=await B(a);return v.state.next({...!$(r)||_.isValid&&n!==t.isValid?{}:{name:r},...s.resolver||!r?{isValid:n}:{},errors:t.errors}),i.shouldFocus&&!d&&fe(a,le,r?c:g.mount),d},Ne=r=>{const i={...y.mount?l:u};return S(r)?i:$(r)?f(i,r):r.map(n=>f(i,n))},Be=(r,i)=>({invalid:!!f((i||t).errors,r),isDirty:!!f((i||t).dirtyFields,r),isTouched:!!f((i||t).touchedFields,r),isValidating:!!f((i||t).validatingFields,r),error:f((i||t).errors,r)}),ht=r=>{r&&ve(r).forEach(i=>C(t.errors,i)),v.state.next({errors:r?t.errors:{}})},Ie=(r,i,n)=>{const d=(f(a,r,{_f:{}})._f||{}).ref;A(t.errors,r,{...i,ref:d}),v.state.next({name:r,errors:t.errors,isValid:!1}),n&&n.shouldFocus&&d&&d.focus&&d.focus()},vt=(r,i)=>X(r)?v.values.subscribe({next:n=>r(x(void 0,i),n)}):x(r,i,!0),De=(r,i={})=>{for(const n of r?ve(r):g.mount)g.mount.delete(n),g.array.delete(n),i.keepValue||(C(a,n),C(l,n)),!i.keepError&&C(t.errors,n),!i.keepDirty&&C(t.dirtyFields,n),!i.keepTouched&&C(t.touchedFields,n),!i.keepIsValidating&&C(t.validatingFields,n),!s.shouldUnregister&&!i.keepDefaultValue&&C(u,n);v.values.next({values:{...l}}),v.state.next({...t,...i.keepDirty?{isDirty:b()}:{}}),!i.keepIsValid&&J()},Pe=({disabled:r,name:i,field:n,fields:d,value:c})=>{if(q(r)){const o=r?void 0:S(c)?Re(n?n._f:f(d,i)._f):c;A(l,i,o),ee(i,o,!1,!1,!0)}},we=(r,i={})=>{let n=f(a,r);const d=q(i.disabled);return A(a,r,{...n||{},_f:{...n&&n._f?n._f:{ref:{name:r}},name:r,mount:!0,...i}}),g.mount.add(r),n?Pe({field:n,disabled:i.disabled,name:r,value:i.value}):E(r,!0,i.value),{...d?{disabled:i.disabled}:{},...s.progressive?{required:!!i.required,min:oe(i.min),max:oe(i.max),minLength:oe(i.minLength),maxLength:oe(i.maxLength),pattern:oe(i.pattern)}:{},name:r,onChange:ue,onBlur:ue,ref:c=>{if(c){we(r,i),n=f(a,r);const o=S(c.value)&&c.querySelectorAll&&c.querySelectorAll("input,select,textarea")[0]||c,h=Ut(o),m=n._f.refs||[];if(h?m.find(L=>L===o):o===n._f.ref)return;A(a,r,{_f:{...n._f,...h?{refs:[...m.filter(Ce),o,...Array.isArray(f(u,r))?[{}]:[]],ref:{type:o.type,name:r}}:{ref:o}}}),E(r,!1,void 0,o)}else n=f(a,r,{}),n._f&&(n._f.mount=!1),(s.shouldUnregister||i.shouldUnregister)&&!(st(g.array,r)&&y.action)&&g.unMount.add(r)}}},je=()=>s.shouldFocusError&&fe(a,le,g.mount),_t=r=>{q(r)&&(v.state.next({disabled:r}),fe(a,(i,n)=>{let d=r;const c=f(a,n);c&&q(c._f.disabled)&&(d||(d=c._f.disabled)),i.disabled=d},0,!1))},qe=(r,i)=>async n=>{let d;n&&(n.preventDefault&&n.preventDefault(),n.persist&&n.persist());let c=U(l);if(v.state.next({isSubmitting:!0}),s.resolver){const{errors:o,values:h}=await Y();t.errors=o,c=h}else await B(a);if(C(t.errors,"root"),N(t.errors)){v.state.next({errors:{}});try{await r(c,n)}catch(o){d=o}}else i&&await i({...t.errors},n),je(),setTimeout(je);if(v.state.next({isSubmitted:!0,isSubmitting:!1,isSubmitSuccessful:N(t.errors)&&!d,submitCount:t.submitCount+1,errors:t.errors}),d)throw d},bt=(r,i={})=>{f(a,r)&&(S(i.defaultValue)?I(r,U(f(u,r))):(I(r,i.defaultValue),A(u,r,U(i.defaultValue))),i.keepTouched||C(t.touchedFields,r),i.keepDirty||(C(t.dirtyFields,r),t.isDirty=i.defaultValue?b(r,U(f(u,r))):b()),i.keepError||(C(t.errors,r),_.isValid&&J()),v.state.next({...t}))},We=(r,i={})=>{const n=r?U(r):u,d=U(n),c=N(r),o=c?u:d;if(i.keepDefaultValues||(u=n),!i.keepValues){if(i.keepDirtyValues)for(const h of g.mount)f(t.dirtyFields,h)?A(o,h,f(l,h)):I(h,f(o,h));else{if(Le&&S(r))for(const h of g.mount){const m=f(a,h);if(m&&m._f){const L=Array.isArray(m._f.refs)?m._f.refs[0]:m._f.ref;if(Ve(L)){const M=L.closest("form");if(M){M.reset();break}}}}a={}}l=e.shouldUnregister?i.keepDefaultValues?U(u):{}:U(o),v.array.next({values:{...o}}),v.values.next({values:{...o}})}g={mount:i.keepDirtyValues?g.mount:new Set,unMount:new Set,array:new Set,watch:new Set,watchAll:!1,focus:""},y.mount=!_.isValid||!!i.keepIsValid||!!i.keepDirtyValues,y.watch=!!e.shouldUnregister,v.state.next({submitCount:i.keepSubmitCount?t.submitCount:0,isDirty:c?!1:i.keepDirty?t.isDirty:!!(i.keepDefaultValues&&!Z(r,u)),isSubmitted:i.keepIsSubmitted?t.isSubmitted:!1,dirtyFields:c?[]:i.keepDirtyValues?i.keepDefaultValues&&l?he(u,l):t.dirtyFields:i.keepDefaultValues&&r?he(u,r):{},touchedFields:i.keepTouched?t.touchedFields:{},errors:i.keepErrors?t.errors:{},isSubmitSuccessful:i.keepIsSubmitSuccessful?t.isSubmitSuccessful:!1,isSubmitting:!1})},He=(r,i)=>We(X(r)?r(l):r,i);return{control:{register:we,unregister:De,getFieldState:Be,handleSubmit:qe,setError:Ie,_executeSchema:Y,_getWatch:x,_getDirty:b,_updateValid:J,_removeUnmounted:ge,_updateFieldArray:F,_updateDisabledField:Pe,_getFieldArray:k,_reset:We,_resetDefaultValues:()=>X(s.defaultValues)&&s.defaultValues().then(r=>{He(r,s.resetOptions),v.state.next({isLoading:!1})}),_updateFormState:r=>{t={...t,...r}},_disableForm:_t,_subjects:v,_proxyFormState:_,_setErrors:K,get _fields(){return a},get _formValues(){return l},get _state(){return y},set _state(r){y=r},get _defaultValues(){return u},get _names(){return g},set _names(r){g=r},get _formState(){return t},set _formState(r){t=r},get _options(){return s},set _options(r){s={...s,...r}}},trigger:ne,register:we,handleSubmit:qe,watch:vt,setValue:I,getValues:Ne,reset:He,resetField:bt,clearErrors:ht,unregister:De,setError:Ie,setFocus:(r,i={})=>{const n=f(a,r),d=n&&n._f;if(d){const c=d.refs?d.refs[0]:d.ref;c.focus&&(c.focus(),i.shouldSelect&&c.select())}},getFieldState:Be}}function Kt(e={}){const s=D.useRef(),t=D.useRef(),[a,u]=D.useState({isDirty:!1,isValidating:!1,isLoading:X(e.defaultValues),isSubmitted:!1,isSubmitting:!1,isSubmitSuccessful:!1,isValid:!1,submitCount:0,dirtyFields:{},touchedFields:{},validatingFields:{},errors:e.errors||{},disabled:e.disabled||!1,defaultValues:X(e.defaultValues)?void 0:e.defaultValues});s.current||(s.current={...jt(e),formState:a});const l=s.current.control;return l._options=e,pe({subject:l._subjects.state,next:y=>{at(y,l._proxyFormState,l._updateFormState,!0)&&u({...l._formState})}}),D.useEffect(()=>l._disableForm(e.disabled),[l,e.disabled]),D.useEffect(()=>{if(l._proxyFormState.isDirty){const y=l._getDirty();y!==a.isDirty&&l._subjects.state.next({isDirty:y})}},[l,a.isDirty]),D.useEffect(()=>{e.values&&!Z(e.values,t.current)?(l._reset(e.values,l._options.resetOptions),t.current=e.values,u(y=>({...y}))):l._resetDefaultValues()},[e.values,l]),D.useEffect(()=>{e.errors&&l._setErrors(e.errors)},[e.errors,l]),D.useEffect(()=>{l._state.mount||(l._updateValid(),l._state.mount=!0),l._state.watch&&(l._state.watch=!1,l._subjects.state.next({...l._formState})),l._removeUnmounted()}),D.useEffect(()=>{e.shouldUnregister&&l._subjects.values.next({values:l._getWatch()})},[e.shouldUnregister,l]),s.current.formState=it(a,l),s.current}const qt=({children:e,className:s,name:t,label:a,errorText:u})=>ae.jsxs("div",{className:s,children:[ae.jsx("label",{htmlFor:t,className:"mb-[8px] text-[0.85rem] md:text-[0.875rem] leading-[1rem] font-deja-vu-sans",children:a}),e,ae.jsx("span",{className:"h-[20px] block text-sm break-words text-error mt-[4px]",children:u})]}),zt=({className:e,control:s,name:t,rules:a,label:u,disabled:l,onFieldChange:y})=>ae.jsx(Rt,{control:s,name:t,rules:a,render:({field:g,fieldState:{error:V}})=>ae.jsx(qt,{className:e,label:u,name:t,errorText:V==null?void 0:V.message,children:ae.jsx(mt,{...g,id:t,className:At(V&&"border-error"),disabled:l,onChange:w=>{g.onChange(w),y==null||y(w.target.value)}})})}),Jt=2,Qt=8,Xt=/[0-9]/,Yt=/[!@#$%^&*()_+{}\[\]:;<>,.?~\-\/\\]/;export{Rt as C,zt as F,Jt as M,Qt as a,Xt as d,Yt as s,Kt as u};