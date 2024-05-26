# handwriting-recognition-service
***
### Требования к ПО:
1. Docker
***
### Как запустить:
1. Запуск:
>```Bash
>chmod +x docker/backend/entrypoint_backend.sh
>chmod +x docker/backend/entrypoint_celery.sh
>docker compose -f .\compose.yml up -d --build
>```
2. Если нет записей в БД - сервер кидает 500 ошибку (или можно проверить через Admined - localhost:8080, данные для входа в бд есть в [compose.yml](/compose.yml)), необходимо выполнить следующие команды:
>```Bash
>docker exec <backend-container> chmod +x setup/config_db.sh
>docker exec <backend-container> setup/config_db.sh
>```
***
### Тестовые пользователи:
| Права доступа | Email | Пароль |
| :-| :- | :- |
| Пользователь | user@gmail.com | user |
| Библиотекарь | librarian@gmail.com | librarian |
| Модератор | moderator@gmail.com | moderator |
| Администратор | admin@gmail.com | admin |
| Суперпользователь | superuser@gmail.com | superuser |
***
### Пути:
* /admin - админ-панель Djnago
* /api/docs - Swagger документация
* [GET] /api/metrics - метрики 
* [GET] /api/demodocs - демонстарционные документы
* [POST] /api/recognize - распознать текст на изображении
