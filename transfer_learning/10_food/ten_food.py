from tensor_tool import image_data, file
from PlotGraph import loss_accuracy
import tensorflow as tf
import tensorflow_hub as hub
from tensor_tool import callbacks

file.check_file("../../Dataset/10_food_classes_10_percent")

train_data, test_data = image_data.get_data("../../Dataset/10_food_classes_10_percent/train",
                                            "../../Dataset/10_food_classes_10_percent/test",
                                            class_mode="categorical")

resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

resnet_feature_extractor_layer = hub.KerasLayer(resnet_url, trainable=False,
                                                name="feature_extraction_layer",
                                                input_shape=(224, 224, 3))

efficientnet_feature_extractor_layer = hub.KerasLayer(efficientnet_url, trainable=False,
                                                      name="feature_extraction_layer",
                                                      input_shape=(224, 224, 3))

model_1 = tf.keras.Sequential([
    resnet_feature_extractor_layer,
    tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
])

model_1.summary()

model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data),
                        callbacks=[callbacks.create_tensorboard_callback(dirname="tensor_hub",
                                                                         experiment_name="resnet_v2_50")])

loss_accuracy.plot_loss_curves(history_1, model_name="resnet_v2_50 ")
loss_accuracy.plot_accuracy_curves(history_1, model_name="resnet_v2_50 ")

model_2 = tf.keras.Sequential([
    efficientnet_feature_extractor_layer,
    tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
])

model_2.summary()

model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_2 = model_2.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data),
                        callbacks=[callbacks.create_tensorboard_callback(dirname="tensor_hub",
                                                                         experiment_name="efficientnet_b0")])

loss_accuracy.plot_loss_curves(history_2, model_name="efficientnet_b0 ")
loss_accuracy.plot_accuracy_curves(history_2, model_name="efficientnet_b0 ")
