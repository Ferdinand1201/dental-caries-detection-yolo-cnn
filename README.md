# Sistem AI explicabil pentru detectarea cariilor dentare folosind YOLOv8 și clasificare CNN

Acest repository conține codul sursă pentru un sistem inteligent de detecție a cariilor dentare, dezvoltat ca parte a lucrării mele de licență. Sistemul utilizează modelul YOLOv8 pentru detectarea automată a zonelor afectate din imagini dentare și un clasificator CNN pentru analiza detaliată a regiunilor decupate, însoțită de explicații vizuale generate prin tehnici de tip XAI (Grad-CAM++ și Integrated Gradients).

## Funcționalități principale

* Detecția cariilor dentare în imagini intraorale, utilizând modelul YOLOv8.

* Clasificarea regiunilor suspecte cu ajutorul unui CNN bazat pe ResNet-18.

* Generarea de explicații vizuale pentru deciziile CNN, folosind metodele Grad-CAM++ și Integrated Gradients.

* Interfață web prietenoasă, care permite încărcarea imaginilor și vizualizarea predicțiilor și explicațiilor.

* Posibilitatea de a mări regiunile decupate și de a vizualiza în detaliu zonele evidențiate de AI
  
## Structura proiectului

### Script-uri și notebook-uri

* `convert.ipynb` – Conversie din formatul Supervisely în format YOLOv8
* `predict.ipynb` – Script pentru testarea și vizualizarea detecției cariilor pe imagini personalizate
* `train.ipynb` – Script de antrenare pentru modelul YOLOv8
* `train_caries_classifier.py` – Script de antrenare a clasificatorului CNN ResNet-18 pe regiunile decupate
* `cnn_explainer.py` – Modul pentru generarea explicațiilor Grad-CAM++ și Integrated Gradients pentru clasificator
* `object_detector.py` – Backend Flask care gestionează logica de detecție, clasificare și explicabilitate
* `index.html` – Interfața web a aplicației

### Modele
  
* `best.pt` – Modelul YOLOv8 antrenat pe 30 de epoci
* `model_cnn.pth` – Modelul CNN antrenat pentru clasificarea imaginilor dentare

### Resurse și configurări

* `README.md` - Documentația proiectului
* `requirements.txt`- lista completă a pachetelor Python necesare

### Foldere auxiliare

* `dataset/` – Setul de date folosit pentru antrenarea CNN-ului (neinclus în proiect)
* `crops/` – Se creează automat la rularea aplicației; conține regiunile decupate de YOLOv8 pentru clasificare
* `explanations/` – Se generează automat; conține hărțile explicative (Grad-CAM++ și Integrated Gradients)
  

## Date de antrenare

Sistemul a fost antrenat pe datasetul DentalAI, disponibil la: https://datasetninja.com/dentalai. Pentru a utiliza acest set de date, este necesară conversia sa în formatul YOLOv8 folosind notebook-ul `convert.ipynb`

## Instrucțiuni de utilizare

1. Asigurați-vă că toate fișierele proiectului se află în același director.
2. Instalați pachetele necesare cu comanda:

<pre>  pip install -r requirements.txt </pre>

3. Porniți aplicația rulând:

<pre> python object_detector.py </pre>


4. Interfața web va fi disponibilă la adresa:
   
   http://localhost:8080

# Despre această versiune

Proiectul include o serie de extinderi realizate în cadrul lucrării de licență:

* Integrarea unui clasificator CNN ResNet18 aplicat pe regiunile decupate de YOLOv8, pentru o clasificare mai precisă.
* Implementarea explicațiilor vizuale folosind tehnicile Grad-CAM++ și Integrated Gradients, pentru a evidenția zonele relevante în procesul de decizie.
* Generarea automată de hărți explicative pentru fiecare regiune detectată, vizibile direct în interfața web.
* Dezvoltarea unei interfețe web moderne, cu suport pentru zoom pe regiuni, mod întunecat (dark mode), validare a fișierelor și afișare organizată a rezultatelor.
* Optimizări de scalabilitate locală: salvare automată a rezultatelor, sortarea logică a regiunilor, modularizarea codului și integrarea completă a fluxului de inferență + explicație.


## Licență

Acest proiect este derivat din [yolov8_caries_detector](https://github.com/andreygermanov/yolov8_caries_detector), publicat sub licența GNU General Public License v3.0 (GPLv3).

Toate modificările și extinderile aduse sunt distribuite tot sub licența **GPLv3**, în conformitate cu termenii acesteia.

Pentru detalii complete, consultați fișierul [LICENSE](./LICENSE).


