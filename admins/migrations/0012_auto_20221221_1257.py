# Generated by Django 3.2.16 on 2022-12-21 05:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('admins', '0011_auto_20221206_1407'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='foto10',
            field=models.FileField(blank=True, null=True, upload_to='dataset/'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='foto4',
            field=models.FileField(blank=True, null=True, upload_to='dataset/'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='foto5',
            field=models.FileField(blank=True, null=True, upload_to='dataset/'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='foto6',
            field=models.FileField(blank=True, null=True, upload_to='dataset/'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='foto7',
            field=models.FileField(blank=True, null=True, upload_to='dataset/'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='foto8',
            field=models.FileField(blank=True, null=True, upload_to='dataset/'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='foto9',
            field=models.FileField(blank=True, null=True, upload_to='dataset/'),
        ),
    ]