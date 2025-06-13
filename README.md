# Sistem AI explicabil pentru detectarea cariilor dentare folosind YOLOv8 și clasificare CNN

Acest repository conține codul sursă pentru un sistem inteligent de detecție a cariilor dentare, dezvoltat ca parte a lucrării mele de licență. Sistemul utilizează modelul YOLOv8 pentru detectarea automată a zonelor afectate din imagini dentare și un clasificator CNN pentru analiza detaliată a regiunilor decupate, însoțită de explicații vizuale generate prin tehnici de tip XAI (Grad-CAM++ și Integrated Gradients).

## Funcționalități principale

* Detecția cariilor dentare în imagini intraorale, utilizând modelul YOLOv8.

* Clasificarea regiunilor suspecte în carie / non-carie cu ajutorul unui CNN bazat pe ResNet-18.

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
  
* `dataset/` – Setul de date folosit pentru antrenarea CNN-ului
* `crops/` – Regiunile decupate automat de YOLOv8, salvate pentru clasificare CNN
* `explanations/` – Hărțile explicative generate (Grad-CAM++ și Integrated Gradients)
* `test_images/` – Imagini folosite pentru testarea funcționalității aplicației

## Date de antrenare

Sistemul a fost antrenat pe datasetul DentalAI, disponibil la: https://datasetninja.com/dentalai. Pentru a utiliza acest set de date, este necesară conversia sa în formatul YOLOv8 folosind notebook-ul `convert.ipynb`


## Instrucțiuni de instalare

* Clonează acest repository
* Instalează dependințele: **pip install -r requirements.txt**
* Asigură-te că fișierele `object_detector.py`, `index.html`, `best.pt` și `model_cnn.pth` sunt în același director
* Rulează serverul local
* 
```
python object_detector.py
```

Aplicația va fi disponibilă la adresa: http://localhost:8080

# Despre această versiune

Acest proiect reprezintă o versiune extinsă a repository-ului original [yolov8_caries_detector](https://github.com/andreygermanov/yolov8_caries_detector), creat de Andrey Germanov.

Contribuțiile proprii aduse includ:

* Integrarea unui clasificator CNN dedicat (ResNet-18) aplicat pe regiunile decupate de YOLOv8, pentru o clasificare binară (carie / non-carie) mai precisă.
* Implementarea explicațiilor vizuale folosind tehnicile Grad-CAM++ și Integrated Gradients, pentru a evidenția zonele relevante în procesul de decizie.
* Generarea automată de hărți explicative pentru fiecare regiune detectată, vizibile direct în interfața web.
* Dezvoltarea unei interfețe web moderne, cu suport pentru zoom pe regiuni, mod întunecat (dark mode), validare a fișierelor și afișare organizată a rezultatelor.
* Optimizări de scalabilitate locală: salvare automată a rezultatelor, sortarea logică a regiunilor, modularizarea codului și integrarea completă a fluxului de inferență + explicație.

Aceste extinderi și îmbunătățiri au fost realizate în cadrul lucrării de licență, sub coordonarea prof. univ. dr. Darian Onchiș, la Universitatea de Vest din Timișoara.

Mulțumiri autorului original pentru codul sursă oferit ca punct de plecare în această cercetare.

## Licență

Acest proiect este derivat din [yolov8_caries_detector](https://github.com/andreygermanov/yolov8_caries_detector), publicat sub licența GNU General Public License v3.0 (GPLv3).

Toate modificările și extinderile aduse – clasificatorul CNN ResNet18, integrarea metodelor explicabile Grad-CAM++ și Integrated Gradients, precum și interfața web interactivă – sunt distribuite tot sub licența **GPLv3**, în conformitate cu termenii acesteia.

> Acest software este oferit cu bună-credință, în speranța că va fi util, dar **fără nicio garanție**, nici explicită, nici implicită – inclusiv fără garanții de vandabilitate sau de adecvare la un anumit scop.

Pentru detalii complete, consultă fișierul [LICENSE](./LICENSE).


