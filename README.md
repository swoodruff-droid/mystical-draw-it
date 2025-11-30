# Mystical Draw It!
This project is a mini QuickDraw-inspired model that recognizes doodles of five mystical subjects.

## Project Setup
📁 Mystical Draw It!    
|____ *training.py*   
|____ *app.py*   
|____ 📁*templates*   
------ |____ </> *index.html*   
|____ *angel.npy*   
|____ *castle.npy*   
|____ *dragon.npy*   
|____ *flying_saucer.npy*   
|____ *mermaid.npy*   
  
## Dependencies
*app.py*   
- Flask   
- TensorFlow   
- NumPy   
- Base64   
- PIL   
- io

*training.py*
- TensorFlow
- Sklearn
- NumPy

## How to Run Code (PyCharm)
1. Download all files, including the 2 python files, template folder with the HTML file,
2. Download the five Quick, Draw! files located at here: https://drive.google.com/drive/folders/1aWEsycj6HlDU_XtRPbFaRfRZYT_OSMVK?usp=sharing
3. Upload everything to your IDE. 
4. Ensure that everything is on the same branch level EXCEPT the HTML file. That file should be one level deeper, in the templates folder.
5. The first file to run is *training.py*. A new file named *"mythical_draw_model.keras"* will be created. 
6. Once training is completed, run *app.py*.
7. Click the link included in the output.
8. Try out the model!!
