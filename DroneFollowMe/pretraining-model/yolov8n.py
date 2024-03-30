from ultralytics import YOLO
import multiprocessing

def main():
    # Load a model
    model = YOLO('./models/best1.pt')  # load a pretrained model (recommended for training)
    
    
    results = model.train(data='VisDrone.yaml', 
                        epochs=50, 
                        imgsz=640, 
                        batch=8) 

    results=model.val()

    success = model.export(format='onnx') 

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()