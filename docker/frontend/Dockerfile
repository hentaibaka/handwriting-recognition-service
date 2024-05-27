FROM node:20 as build
WORKDIR /usr/src/app
ENV PATH /usr/src/app/node_modules/.bin:$PATH
COPY frontend/ .
RUN npm install 
RUN npm run build

FROM nginx:stable-alpine
COPY --from=build /usr/src/app/dist /usr/share/nginx/html

CMD ["nginx", "-g", "daemon off;"]