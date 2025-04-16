from data_utils import download_dataset, load_data
from preprocess import process_dataframe
from models import cnn_model
from visualization import plot_sample_images
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

def main():
    data_root = download_dataset()
    train_df = load_data(data_root/"train", "train")
    test_df  = load_data(data_root/"test",  "test")
    val_df   = load_data(data_root/"valid", "validation")

    le = LabelEncoder()
    X_train, y_train = process_dataframe(train_df, le, fit_encoder=True)
    X_test,  y_test  = process_dataframe(test_df,  le)
    X_val,   y_val   = process_dataframe(val_df,   le)

    # plot_sample_images(train_df)

    datagen = ImageDataGenerator(
        rotation_range=6, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1
    )
    datagen.fit(X_train)

    model = cnn_model(num_classes=len(le.classes_))
    early = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=60,
        validation_data=(X_test, y_test),
        callbacks=[early]
    )
    model.save("cnn_model.h5")

if __name__ == "__main__":
    main()
