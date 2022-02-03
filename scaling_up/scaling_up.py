from tensor_tool import image_data, file, callbacks
from plot_graph import loss_accuracy, image_data_graph
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

file.check_file("../Dataset/101_food_classes_10_percent")

train_data_10, test_data_10 = image_data.get_dataset("../Dataset/101_food_classes_10_percent/train",
                                                     "../Dataset/101_food_classes_10_percent/test",
                                                     class_mode="categorical")

tf.random.set_seed(42)

# base_model = tf.keras.applications.EfficientNetB0(include_top=False)
# base_model.trainable = False
# inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
# x = image_data.augmentation_layer()(inputs)
# x = base_model(x, training=False)
# x = tf.keras.layers.GlobalAvgPool2D(name="global_avg_pool_2D")(x)
# outputs = tf.keras.layers.Dense(101, activation="softmax", name="output_layer")(x)
#
# model_0 = tf.keras.Model(inputs, outputs)
#
# model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#                 optimizer=tf.keras.optimizers.Adam(),
#                 metrics=["accuracy"])
#
# history_base = model_0.fit(train_data_10,
#                            epochs=5,
#                            steps_per_epoch=len(train_data_10),
#                            validation_data=test_data_10,
#                            validation_steps=int(0.5 * len(test_data_10)),
#                            callbacks=[callbacks.create_tensorboard_callback("food_101", "BaseLine"),
#                                       callbacks.create_model_checkpoint("baseline_checkpoint")])
#
# result_base = model_0.evaluate(test_data_10)
#
# loss_accuracy.plot_loss_curves(history_base, "model_0 ")
# loss_accuracy.plot_accuracy_curves(history_base, "model_0 ")
#
# # model_1
# model_0.load_weights("baseline_checkpoint/checkpoint.ckpt")
#
# base_model.trainable = True
# for layer in base_model.layers[:-10]:
#     layer.trainable = False
#
# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name, layer.trainable)
#
# model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                 metrics=["accuracy"])
#
# history_1 = model_0.fit(train_data_10,
#                         epochs=10,
#                         initial_epoch=history_base.epoch[-1],
#                         steps_per_epoch=len(train_data_10),
#                         validation_data=test_data_10,
#                         validation_steps=int(0.5 * len(test_data_10)),
#                         callbacks=[callbacks.create_tensorboard_callback("food_101", "model_1"),
#                                    callbacks.create_model_checkpoint("model_1_checkpoint")])
#
# result_1 = model_0.evaluate(test_data_10)
#
# loss_accuracy.plot_loss_curves(history_1, "model_1 ")
# loss_accuracy.plot_accuracy_curves(history_1, "model_1 ")
#
# print(f"Model 0: {result_base}")
# print(f"Model 1: {result_1}")
#
# model_0.save("model_1", save_format='h5')
# loaded_model = tf.keras.models.load_model("model_1")
# result_loaded = loaded_model.evaluate(test_data_10)
# print(f"Model Load: {result_loaded}")

downloaded_model = tf.keras.models.load_model("./06_101_food_class_10_percent_saved_big_dog_model")
result_downloaded = downloaded_model.evaluate(test_data_10)
print(f"Model Download: {result_downloaded}")

# make y_labels
pred_probs = downloaded_model.predict(test_data_10, verbose=1)

y_labels, pred_classes = image_data.create_y_labels_y_pred(pred_probs=pred_probs, test_data=test_data_10)

image_data_graph.plot_confusion_matrix(y_true=y_labels, y_preds=pred_classes, classes=test_data_10.class_names,
                                       figsize=(100, 100), text_size=20)

image_data_graph.plot_classification_report(y_true=y_labels, y_pred=pred_classes, class_names=test_data_10.class_names)

image_data.predict_with_model("../Dataset/101_foods_pred/chicken_wings.jpeg", downloaded_model, test_data_10.class_names, False)

image_data.predict_random_data(downloaded_model, "../Dataset/101_food_classes_10_percent/test",
                               test_data_10.class_names)
