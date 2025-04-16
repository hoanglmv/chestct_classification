# train_resnet34.py
from data_utils import download_dataset, load_data
from preprocess import process_dataframe
from models import resnet34_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from resnet34 import resnet34_model
def main():
    # 1) Download & load data
    data_root = download_dataset()
    train_df = load_data(data_root/"train", "train")
    test_df  = load_data(data_root/"test",  "test")
    val_df   = load_data(data_root/"valid", "validation")

    # 2) Preprocess
    le = LabelEncoder()
    X_train, y_train = process_dataframe(train_df, le, fit_encoder=True)
    X_test,  y_test  = process_dataframe(test_df,  le)
    X_val,   y_val   = process_dataframe(val_df,   le)

    # 3) Build model
    model = resnet34_model(
        input_shape=X_train.shape[1:], 
        num_classes=len(le.classes_),
        learning_rate=1e-4
    )

    # 4) Callbacks
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # 5) Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=50,
        callbacks=[early]
    )

    # 6) Lưu model
    model.save("resnet34_model.h5")

if __name__ == "__main__":
    main()
