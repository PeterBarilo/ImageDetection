from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath(os.path.join(execution_path, "[MODEL]")
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path,join(execution_path))
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction," : ", eachProbability)

