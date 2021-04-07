# coding=utf-8

from flask_login import UserMixin


# user models
class User(UserMixin):
    id = None
    grade = None
    group = None
    last_login = None
    last_logout = None
    online_time = None


class Teacher(UserMixin):
    id = None
    last_login = None
    last_logout = None
