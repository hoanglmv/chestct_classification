from tensorflow.keras.models import load_model
from data_utils import download_dataset, load_data
from preprocess import process_dataframe
from visualization import plot_predictions
from sklearn.preprocessing import LabelEncoder

def main():
    data_root = download_dataset()
    val_df  = load_data(data_root/"valid", "validation")
    test_df = load_data(data_root/"test",  "test")

    le = LabelEncoder()
    # fit tạm để có đầy đủ classes
    _ = le.fit(list(val_df["label"]) + list(test_df["label"]))

    X_val, y_val   = process_dataframe(val_df,  le)
    X_test, y_test = process_dataframe(test_df, le)

    for fname in ["cnn_model.h5", "vgg16_model.h5"]:
        print(f"== Đánh giá {fname} ==")
        model = load_model(fname)
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Acc: {acc:.4f}, Loss: {loss:.4f}")
        plot_predictions(model, X_val, y_val, le)

if __name__ == "__main__":
    main()
