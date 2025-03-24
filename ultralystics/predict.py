from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('C:\Python\PythonProject\catdog\ultralystics\runs\detect\train\weights\best.pt')

# 使用模型进行预测
results = model.predict(source='C:\Python\PythonProject\yolo\dogcat/videos', save=True)