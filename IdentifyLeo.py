from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd()


prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
prediction.loadModel()

print('-- DenseNet')
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "eevee1.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)


prediction = ImagePrediction()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

print('\n\n-- InceptionV3')
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "eevee1.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)