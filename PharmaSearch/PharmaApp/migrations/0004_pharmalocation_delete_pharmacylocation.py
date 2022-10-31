# Generated by Django 4.1.2 on 2022-10-27 22:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PharmaApp', '0003_alter_pharmacylocation_pharmaname'),
    ]

    operations = [
        migrations.CreateModel(
            name='pharmaLocation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pharmaName', models.CharField(blank=True, max_length=255, null=True)),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
            ],
        ),
        migrations.DeleteModel(
            name='pharmacyLocation',
        ),
    ]