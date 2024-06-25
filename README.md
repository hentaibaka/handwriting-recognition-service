# handwriting-recognition-service
***
### Требования к ПО:
1. Docker
***
### Как запустить:
1. Запуск:
>```Bash
>docker compose -f .\compose.yml up -d --build
>```
2. Скачть [TrOCR model]() и поместить папку trocr в [backend/recognition_module/models/model/](/backend/recognition_module/models/model).
3. Если нет записей в БД - сервер кидает 500 ошибку (или можно проверить через Admined - host:8080, данные для входа в бд есть в [.env.prod](/docker/.env.prod)), необходимо выполнить следующие команды:
>```Bash
>docker compose -f compose.yml exec backend setup/config_db.sh
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
* /api/docs - Swagger документация\
* /graphana - Graphana (admin:admin для входа)
***
### Инструкция:
1. Как пользоваться сервисом описано в [инструкции](/docs/base.md)
2. Действия, которые может выполнять обычный пользователь описаны [тут](/docs/user.md)
3. Действия, которые может выполнять библиотекарь описаны [тут](/docs/librarian.md)
4. Действия, которые может выполнять модератор описаны [тут](/docs/moderator.md)
5. Действия, которые может выполнять администратор описаны [тут](/docs/admin.md)
***
