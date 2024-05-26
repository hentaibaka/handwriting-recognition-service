#!/bin/sh

#bash -c "python3 manage.py migrate && python3 manage.py loaddata metrics groups && python3 manage.py shell < setup/adminuser.py python3 manage.py shell < setup/permissions.py && python3 manage.py shell < setup/testdata.py"

python3 manage.py migrate
python3 manage.py loaddata metrics groups
python3 manage.py shell < setup/permissions.py
python3 manage.py shell < setup/adminuser.py
python3 manage.py shell < setup/testdata.py