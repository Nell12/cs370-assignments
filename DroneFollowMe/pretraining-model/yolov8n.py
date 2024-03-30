from ultralytics import YOLO
import multiprocessing

def main():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    
    #Set parameters for training
    results = model.train(data='VisDrone.yaml', 
                        epochs=50, 
                        imgsz=640, 
                        batch=8) 
    
    #Validate data
    results=model.val()

    success = model.export(format='onnx') 

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()