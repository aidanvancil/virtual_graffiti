# Generated by Django 4.2.7 on 2024-05-03 06:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Laser',
            fields=[
                ('id', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('color', models.CharField(default='#777777', max_length=30, null=True)),
                ('size', models.IntegerField(default=10)),
            ],
        ),
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(max_length=30)),
                ('last_name', models.CharField(max_length=30)),
                ('laser', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='app.laser')),
            ],
        ),
    ]
