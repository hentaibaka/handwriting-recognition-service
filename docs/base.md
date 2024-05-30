# Инструкция: начало
***
1. Вход в аккаунт:\
Первое, что нужно сделать, это войти в аккаунт. Сделать это можно по [этой ссылке](http://45.12.230.37/admin/login/?next=/admin/). Вводим почту и пароль от аккаунта. Данные для входа можно взять в [README](https://github.com/hentaibaka/handwriting-recognition-service/blob/main/README.md)\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/14af3420-8a90-44bf-9b14-e47d41567165)
2. Восстановление пароля:\
Если вы забыли пароль, находим ссылку "Забыли пароль?".\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/410bb689-a291-459b-87c8-ac1e8dad4996)\
Переходим по ней. Вас перекинет на форму восстановления пароля. Введите почту на которую зарегистрирован ваш аккаунт. На эту почту придёт письмо с ссылкой для восстановление пароля.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/ba2707c2-505c-4cd9-b51c-fc1b08855018)\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/4a91df19-ff30-4873-b16e-b4fd40d6ebb3)\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/63cdde6f-637c-486a-89ac-f4f114e46b31)\
Перейдя по ссылке в письме, укажите новый пароль.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/6c190f4f-6607-4e9c-92fa-c3b1ec67e752)\
3. Интерфейс:\
Войдя в аккаунт, вы окажитель на главной странице админ-панели.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/deb68c1c-8255-4ef7-8ee9-10ce5afbb118)\
Кратко пройдёмся по интерфейсу.\
Это таблицы к которым вы имеете доступ, если нажать на название таблицы откроется окно просмотра записей. Справа от названия таблицы перечислены действия, которые вы можете совершать с записями в данной таблице, если действий нет, значит вы можете только просматривать записи.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/4f437ea8-d04c-4e83-8f84-8bf6d2e1c673)\
Если открыть любую таблицу, вы увидете примерно следующий интерфейс.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/b1e059be-d3ce-4695-8a72-eddc264c92e4)\
Где, слева вы увидете список других таблиц.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/d3b4528c-bbe1-4afb-a0e1-b7240f005f4d)\
Справа будет панель фильтрации записей по различным параметрам.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/1acb4fdb-83a3-4baa-a95c-fd708b2d4b5c)\
По центру будут записи, содержащиеся в данной таблице. Сверху находится поле для поиска записей по содержимому.
Под ним список массовых действий, которые можно применить сразу к нескольким выбранным записям.
Сама запись может иметь действия или поля которые можно совершить или изменить в этой записи не переходя в окно редактирования этой записи. Чтобы перейте в окно изменения контретной записи нужно нажать на ID данной записи.
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/eb0158b5-9664-4c33-a910-da0197d65b95)\
Перейдя в окно изменения конкретной записи вы увидете похожий интерфейс. Тут можно внести изменения в поля записи и сохранить запись, либо её удалить.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/072d3cd7-0e5d-4d9b-b4dd-0e3962dbb857)
Левее от таблиц на главной странице представлены ваши последние действия.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/df3cc2b4-9b43-4798-861b-2182cbbb1e20)\
В шапке сайта находятся 3 ссылки: изменить свой профиль, поменять пароль и выйти.\
Если открыть страницу изменения профиля, вы сможете поменять личную информацию и некоторые служебные поля (зависит от вашего уровня допуска).\
Почту и ФИО могут поменять все пользователи, поля "активен", "сотрудник" и "группы пользователя" можут изменять только пользователи с группой "администратор".\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/94639641-240e-4e76-94a5-83f00d8b1a40)\
Если открыть изменение пароля, вам нужно будет ввести старый пароль и новый.\
![image](https://github.com/hentaibaka/handwriting-recognition-service/assets/61946499/88d3b6f6-ef63-416b-8e2f-6742966ad35c)
Если выйти из аккаунта вас перенаправит на страницу входа.\
***
На этом базовая инструкция по работе с админ-панелью завершена.\
Далее рекомендуется изучить инструкции для пользователей с конкретной группой прав:
1. Действия, которые может выполнять обычный пользователь описаны [тут](/docs/user.md)
2. Действия, которые может выполнять библиотекарь описаны [тут](/docs/librarian.md)
3. Действия, которые может выполнять модератор описаны [тут](/docs/moderator.md)
4. Действия, которые может выполнять администратор описаны [тут](/docs/admin.md)
