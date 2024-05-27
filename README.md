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
2. Скачть [yolo_model](https://drive.google.com/file/d/1I56uvn7kAMZIh7AZfDIIwjAysJj_sdu7/view) и поместить её в [backend/recognition_module/models/](/backend/recognition_module/models).
3. Удалить [.gitkeep](/backend/static/.gitkeep) из [backend/static/](/backend/static). ```python manage.py collectstatic``` выдаст предупреждение что папка не пустая и из-за этого могут быть проблемы при сборке образа.
4. Если нет записей в БД - сервер кидает 500 ошибку (или можно проверить через Admined - host:8080, данные для входа в бд есть в [.env.prod](/docker/.env.prod)), необходимо выполнить следующие команды:
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
* host:8080/ - Adminer, интерфейс для БД, данные для входа есть в [.env.prod](/docker/.env.prod)
* host:15672/ - RabbitMQ, интерфейс брокера сообщений, данные для входа есть в [.env.prod](/docker/.env.prod)
