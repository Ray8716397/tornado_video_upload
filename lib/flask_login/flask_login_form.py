# coding=utf-8

from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, PasswordField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    user_id = StringField('user_id', validators=[DataRequired()])
    user_password = PasswordField('user_password', validators=[DataRequired()])
    remember_me = BooleanField('remember_me', default=False)
