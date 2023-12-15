from ultralytics import YOLO
 
model = YOLO('F:/Git Repos/Bird-Classification/src/bird-cls.pt')

results = model('F:/Git Repos/Bird-Classification/images/222.jpg')

preds = []
for result in results:
    probs = result.probs
    class_index = probs.top1
    class_name = result.names[class_index]
    score = float(probs.top1conf.cpu().numpy())

print(f"{class_name}, {round(score * 100, 2)}%")