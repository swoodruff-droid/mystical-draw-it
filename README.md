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
1. Download all files, including the 2 python files, template folder with the HTML file, and five .npy files from the Quick, Draw! dataset.
2. Ensure that everything is on the same branch level EXCEPT the HTML file. That file should be one level deeper, in the templates folder.
3. The first file to run is *training.py*. A new file named *"mythical_draw_model.keras"* will be created. 
4. Once training is completed, run *app.py*.
5. Click the link included in the output.
6. Try out the model!!
