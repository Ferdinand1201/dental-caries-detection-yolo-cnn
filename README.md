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

Sistemul a fost antrenat pe datasetul DentalAI, disponibil la: https://datasetninja.com/dentalai. Pentru a utiliza acest set de date, este necesară conversia sa în formatul YOLOv8 folosind notebook-ul convert.ipynb


## Instrucțiuni de instalare

* Clonează acest repository
* Instalează dependințele: **pip install -r requirements.txt**

## Pornește serverul Flask

* Asigură-te că fișierele object_detector.py, index.html, best.pt și model_cnn.pth sunt în același director
* Rulează

```
python object_detector.py
```

Aplicația va fi disponibilă la adresa: http://localhost:8080

# Despre această versiune

Acest proiect reprezintă o versiune extinsă a repository-ului original  https://github.com/andreygermanov/yolov8_caries_detector, creat de Andrey Germanov.

Modificările și contribuțiile proprii includ:

* Integrarea unui clasificator CNN separat (ResNet-18) pentru analiza zonelor decupate.

* Implementarea explicațiilor vizuale prin Grad-CAM++ și Integrated Gradients.

* Generarea de imagini explicative pentru fiecare regiune detectată, vizibile direct în interfața web.

* Dezvoltarea unei interfețe web moderne și interactive, cu funcționalități de zoom, mod întunecat și validare a fișierelor.

* Optimizări pentru scalabilitate locală, salvare automată a rezultatelor, sortarea regiunilor detectate și afișare organizată.

Toate aceste îmbunătățiri au fost realizate în cadrul lucrării de licență, sub coordonarea prof. univ .dr. Darian Onchiș.

Mulțumiri autorului original pentru codul de bază folosit ca punct de plecare în această cercetare.
