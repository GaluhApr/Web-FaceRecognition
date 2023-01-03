# Generated by Django 3.2.16 on 2022-11-29 04:18

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('admins', '0004_alter_absensi_idabsen_alter_absensi_keterangan_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Absen',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tanggal', models.DateField()),
                ('status', models.TextField(choices=[('Masuk', 'Masuk'), ('Ijin', 'Ijin'), ('Sakit', 'Sakit'), ('Alpha', 'Alpha')])),
            ],
            options={
                'db_table': 'tb_absen',
            },
        ),
        migrations.CreateModel(
            name='Mahasiswa',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nim', models.CharField(max_length=10)),
                ('foto', models.ImageField(blank=True, null=True, upload_to='upload/')),
                ('nama', models.CharField(max_length=20)),
                ('semester', models.TextField(choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'), ('7', '7'), ('8', '8')])),
                ('telepon', models.CharField(max_length=14)),
                ('alamat', models.TextField(max_length=20)),
                ('jenisKelamin', models.TextField(choices=[('Laki-laki', 'Laki-laki'), ('Perempuan', 'Perempuan')])),
            ],
            options={
                'db_table': 'tb_mahasiswa',
            },
        ),
        migrations.CreateModel(
            name='Users',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=30)),
                ('password', models.CharField(max_length=40)),
            ],
            options={
                'db_table': 'tb_users',
            },
        ),
        migrations.RenameModel(
            old_name='mataKuliah',
            new_name='Matkul',
        ),
        migrations.DeleteModel(
            name='absensi',
        ),
        migrations.DeleteModel(
            name='beritaAcara',
        ),
        migrations.DeleteModel(
            name='foto',
        ),
        migrations.DeleteModel(
            name='Member',
        ),
        migrations.RenameField(
            model_name='golongan',
            old_name='namaGol',
            new_name='golongan',
        ),
        migrations.RemoveField(
            model_name='golongan',
            name='idGol',
        ),
        migrations.RemoveField(
            model_name='jadwal',
            name='idJadwal',
        ),
        migrations.AddField(
            model_name='jadwal',
            name='golongan',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='admins.golongan'),
        ),
        migrations.AddField(
            model_name='jadwal',
            name='namaDosen',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='admins.dosen'),
        ),
        migrations.AlterField(
            model_name='dosen',
            name='namaDosen',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='dosen',
            name='nip',
            field=models.CharField(max_length=20),
        ),
        migrations.AlterField(
            model_name='jadwal',
            name='jamMulai',
            field=models.TimeField(),
        ),
        migrations.AlterField(
            model_name='jadwal',
            name='jamSelesai',
            field=models.TimeField(),
        ),
        migrations.AlterModelTable(
            name='dosen',
            table='tb_dosen',
        ),
        migrations.AlterModelTable(
            name='golongan',
            table='tb_golongan',
        ),
        migrations.AlterModelTable(
            name='jadwal',
            table='tb_jadwal',
        ),
        migrations.AlterModelTable(
            name='matkul',
            table='tb_matkul',
        ),
        migrations.AddField(
            model_name='mahasiswa',
            name='golongan',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='admins.golongan'),
        ),
        migrations.AddField(
            model_name='absen',
            name='jadwal',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='admins.jadwal'),
        ),
        migrations.AddField(
            model_name='absen',
            name='mahasiswa',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='admins.mahasiswa'),
        ),
    ]
