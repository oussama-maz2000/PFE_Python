# Generated by Django 4.1.2 on 2022-10-28 19:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PharmaApp', '0004_pharmalocation_delete_pharmacylocation'),
    ]

    operations = [
        migrations.AddField(
            model_name='pharmalocation',
            name='cluster',
            field=models.IntegerField(default=-1),
            preserve_default=False,
        ),
    ]
