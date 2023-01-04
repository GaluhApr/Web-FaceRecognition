from django.shortcuts import render, redirect, HttpResponse
from .models import Mahasiswa, Matkul, Dosen, Jadwal
from .forms import Memberform, matakuliahform, dosenform, jadwalform, UserSelection
from django.core.files.storage import FileSystemStorage
from PIL import Image
# from keras.models import load_model
# import numpy as np
# from numpy import asarray
# from numpy import expand_dims
from .resource import jadwaltable, mahasiswatable, dosentable, matkultable
import pickle
import cv2
from .functions import handle_uploaded_file
from django.contrib.auth import authenticate, login, logout
from django.conf import settings
from django.contrib import messages
import numpy as np
import os
from django.http.response import StreamingHttpResponse

BASE_DIR = getattr(settings, 'BASE_DIR')

# Create your views here.
def addvideo_stream(request):
    return StreamingHttpResponse(create_dataset(request),content_type='multipart/x-mixed-replace; boundary=frame')

def video_stream(request):
    
    return StreamingHttpResponse(detect(request),content_type='multipart/x-mixed-replace; boundary=frame')
		
def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(BASE_DIR + '/ml/recognizer/trainer.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            getId, conf = rec.predict(gray[y:y + h, x:x + w])  # This will predict the id of the face
            print(getId, conf)
            confidence = "  {0}%".format(round(100 - conf))
            # print conf;
            if conf < 35:
                try:
                    user = Mahasiswa.objects.get(id=getId)
                except  Mahasiswa.DoesNotExist:
                    pass

                print("User Name", user.nama)

                userId = getId
                if user.nama:
                    cv2.putText(img, user.nama, (x+5, y+h-10), font, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Detected", (x, y + h), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y + h), font, 1, (0, 0, 255), 2)

            cv2.putText(img, str(confidence), (x + 5, y - 5), font, 1, (255, 255, 0), 1)
            
            # Printing that number below the face
            # @Prams cam image, id, location,font style, color, stroke

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        if (cv2.waitKey(1) == ord('q')):
            break
        #elif (userId != 0):
        #    cv2.waitKey(1000)
        #    cam.release()
        #    cv2.destroyAllWindows()
        #    return redirect('/records/details/' + str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('attendance')


def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''


    # Path for face image database
    path = BASE_DIR + '/ml/dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(BASE_DIR+"/ml/haarcascade_frontalface_default.xml");  # function to get the images and label data

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            #faces = detector.detectMultiScale(img_numpy)
            #for (x, y, w, h) in faces:
            #    faceSamples.append(img_numpy[y:y + h, x:x + w])
            #    ids.append(id)
            faceSamples.append(img_numpy)
            ids.append(id)
            # print ID
            cv2.imshow("training", img_numpy)
            cv2.waitKey(10)
        return np.array(faceSamples), np.array(ids)
        #return faceSamples, ids

    print("[INFO] Training faces. It will take a few seconds. Wait ...")

    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, ids)  # Save the model into trainer/trainer.yml
    recognizer.save(BASE_DIR+'/ml/recognizer/trainer.yml')  # Print the numer of faces trained and end program
    print("[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    cv2.destroyAllWindows()
    messages.success(request, "{0} faces trained successfully".format(len(np.unique(ids))) )

    return redirect('listuser')

def create_dataset(request):
    if request.method == "POST":
        face_id = int(request.POST['selected_user'])
        #print("Face ID->", face_id, type(face_id))

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        face_detector = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')  # For each person, enter one numeric face id
        print("[INFO] Initializing face capture. Look the camera and wait ...")  # Initialize individual sampling face count
        count = 0
        while (True):
            ret, img = cam.read()
            # img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            # Skip the process if multiple faces detected:
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    # Save the captured image into the datasets folder
                    cv2.imwrite(BASE_DIR+"/ml/dataset/User." + str(face_id) + '.' +
                                str(count) + ".jpg", gray[y:y + h, x:x + w])
                    cv2.waitKey(250)

                cv2.imshow('Face', img)
                k = cv2.waitKey(1) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 30:  # Take 30 face sample and stop video
                    break  # Do a bit of cleanup
                print(count)
            else:
                print("\n multiple faces detected")

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

        messages.success(request, 'Face successfully registered.')

    else:
        print("Its a GET method.")

    return redirect('listuser')

def signout(request):
    logout(request)
    return redirect('/')
    

def index(request):

    return render(request, 'dashboard.html')


def attendance(request):
    return render(request, 'attendance.html')

# user


def user(request):
    members = Mahasiswa.objects.all()
    context = {
        'Members': members,
        'memberform': Memberform(),
        'imageform': UserSelection(),
        
    }

    return render(request, 'user.html', context,)


def createmember(request):
    if request.method == 'POST':
        form = Memberform(request.POST)
        if form.is_valid():
            form.save()
            return redirect('listuser')
    return redirect('listuser')

# def addimage(request):
#     if request.method == 'POST':
#         form = imageform(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             return redirect('listuser')
#     return redirect('listuser')
        
def delete_member(request, id):
    delmember = Mahasiswa.objects.get(id=id)
    delmember.delete()
    members = Mahasiswa.objects.all()
    return redirect('listuser')


def edit_member(request, id):
    member_edit = Mahasiswa.objects.get(id=id)

    data = {
        'nim': member_edit.nim,
        'nama': member_edit.nama,
        'golongan': member_edit.golongan,
        'semester': member_edit.semester,
        'telepon': member_edit.telepon,
        'alamat': member_edit.alamat,
        'jenisKelamin': member_edit.jenisKelamin,
    }

    admins_member = Memberform(
        request.POST or None, initial=data, instance=member_edit)

    if request.method == 'POST':
        if admins_member.is_valid():
            admins_member.save()
            return redirect('listuser')

    context = {
        'page_title': 'Mahasiswa',
        'Member': member_edit,
        'form': Memberform(initial=data, instance=member_edit)
    }
    return render(request, 'editdata.html', context)


def createdosen(request):
    nip = request.POST["nip"]
    namaDosen = request.POST["namaDosen"]

    admins_dosen = Dosen(nip=nip, namaDosen=namaDosen)
    admins_dosen.save()
    return redirect('listdosen')


def delete_dosen(request, id):
    deldosen = Dosen.objects.get(id=id)
    deldosen.delete()
    deldosen = Dosen.objects.all()
    return redirect('listdosen')


def edit_dosen(request, id):
    dosen_edit = Dosen.objects.get(id=id)

    data = {
        'nip': dosen_edit.nip,
        'namaDosen': dosen_edit.namaDosen,
    }

    admins_dosen = dosenform(request.POST or None,
                             initial=data, instance=dosen_edit)

    if request.method == 'POST':
        if admins_dosen.is_valid():
            admins_dosen.save()
            return redirect('listdosen')

    context = {
        'page_title': 'Dosen',
        'dosen': dosen_edit,
        'form': dosenform(initial=data, instance=dosen_edit)
    }
    return render(request, 'editdata.html', context)


def dosenview(request):
    dosens = Dosen.objects.all()
    context = {
        'Dosens': dosens,
        'form': dosenform()
    }
    return render(request, 'dosen.html', context,)

# matakuliah


def creatematkul(request):
    kodeMK = request.POST["kodeMK"]
    mataKuliah = request.POST["mataKuliah"]
    sks = request.POST["sks"]

    admins_matakuliah = Matkul(kodeMK=kodeMK, mataKuliah=mataKuliah, sks=sks)
    admins_matakuliah.save()
    return redirect('listmatkul')


def edit_matkul(request, id):
    matkul_edit = Matkul.objects.get(id=id)

    data = {
        'kodeMK': matkul_edit.kodeMK,
        'mataKuliah': matkul_edit.mataKuliah,
        'sks': matkul_edit.sks,
    }

    admins_matakuliah = matakuliahform(
        request.POST or None, initial=data, instance=matkul_edit)

    if request.method == 'POST':
        if admins_matakuliah.is_valid():
            admins_matakuliah.save()
            return redirect('listmatkul')

    context = {
        'page_title': 'Matkul',
        'matakuliah': matkul_edit,
        'form': matakuliahform(initial=data, instance=matkul_edit)
    }
    return render(request, 'editdata.html', context)


def delete_matakuliah(request, id):
    deletematakuliah = Matkul.objects.get(id=id)
    deletematakuliah.delete()
    return redirect('listmatkul')


def matkulview(request):
    matkuls = Matkul.objects.all()
    context = {
        'Matkuls': matkuls,
        'form': matakuliahform()
    }
    return render(request, 'matakuliah.html', context,)

# sudah absen


def sudahabsen(request):
    return render(request, 'sudahabsen.html')

# belum absen


def tidakabsen(request):
    return render(request, 'tidakabsen.html')

# absen


def screen(request):
    return render(request, 'attendancescreen.html')


# jadwal
def jadwal(request):
    jadwals = Jadwal.objects.all()
    context = {
        'Jadwals': jadwals,
        'form': jadwalform()
    }
    return render(request, 'jadwal.html', context)


def delete_jadwal(request, id):
    deletejadwal = Jadwal.objects.get(id=id)
    deletejadwal.delete()
    return redirect('listjadwal')


def createjadwal(request):
    namaDosen = request.POST["namaDosen"]
    golongan = request.POST["golongan"]
    matkul = request.POST["matkul"]
    ruangan = request.POST["ruangan"]
    hari = request.POST["hari"]
    jamMulai = request.POST["jamMulai"]
    jamSelesai = request.POST["jamSelesai"]

    admins_jadwal = Jadwal(namaDosen_id=namaDosen, golongan_id=golongan, matkul_id=matkul,
                           ruangan=ruangan, hari=hari, jamMulai=jamMulai, jamSelesai=jamSelesai)
    admins_jadwal.save()
    return redirect('listjadwal')


def edit_jadwal(request, id):
    jadwal_edit = Jadwal.objects.get(id=id)

    data = {
        'namaDosen': jadwal_edit.namaDosen,
        'golongan': jadwal_edit.golongan,
        'matkul': jadwal_edit.matkul,
        'ruangan': jadwal_edit.ruangan,
        'hari': jadwal_edit.hari,
        'jamMulai': jadwal_edit.jamMulai,
        'jamSelesai': jadwal_edit.jamSelesai,

    }

    admins_jadwal = jadwalform(
        request.POST or None, initial=data, instance=jadwal_edit)

    if request.method == 'POST':
        if admins_jadwal.is_valid():
            admins_jadwal.save()
            return redirect('listjadwal')

    context = {
        'page_title': 'Jadwal',
        'jadwal': jadwal_edit,
        'form': jadwalform(initial=data, instance=jadwal_edit)
    }
    return render(request, 'editdata.html', context)


def exportjadwal(request):
    jadwal = jadwaltable()
    dataset = jadwal.export()
    response = HttpResponse(
        dataset.xls, content_type='application/vnd.ms-excel')  # type: ignore
    response['Content-Disposition'] = 'attachment; filename="jadwal.xls"'
    return response


def exportmahasiswa(request):
    mahasiswa = mahasiswatable()
    dataset = mahasiswa.export()
    response = HttpResponse(
        dataset.xls, content_type='application/vnd.ms-excel')  # type: ignore
    response['Content-Disposition'] = 'attachment; filename="mahasiswa.xls"'
    return response


def exportdosen(request):
    dosen = dosentable()
    dataset = dosen.export()
    response = HttpResponse(
        dataset.xls, content_type='application/vnd.ms-excel')  # type: ignore
    response['Content-Disposition'] = 'attachment; filename="dosen.xls"'
    return response


def exportmatkul(request):
    matkul = matkultable()
    dataset = matkul.export()
    response = HttpResponse(
        dataset.xls, content_type='application/vnd.ms-excel')  # type: ignore
    response['Content-Disposition'] = 'attachment; filename="matakuliah.xls"'
    return response
