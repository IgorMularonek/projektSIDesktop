import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QLabel, QPushButton
from model import load_model, load_label_encoders, make_prediction

class PredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predykcja Satysfakcji Klienta")
        self.setGeometry(100, 100, 400, 500)
        self.model = load_model("random_forest")  # Możesz też użyć np. "knn"
        self.encoders = load_label_encoders()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.result_label = QLabel("Wprowadź dane i kliknij przycisk.")
        layout.addWidget(self.result_label)

        self.inputs = {}
        fields = [
            ("gender", "Płeć (Male/Female)"),
            ("customer type", "Typ klienta (Loyal/Disloyal)"),
            ("type of travel", "Typ podróży (Business/Personal)"),
            ("class", "Klasa (Economy/Business/...)"),
            ("age", "Wiek (np. 45)"),
            ("flight distance", "Dystans lotu (np. 1200)")
        ]

        for key, placeholder in fields:
            line = QLineEdit()
            line.setPlaceholderText(placeholder)
            layout.addWidget(line)
            self.inputs[key] = line

        self.predict_button = QPushButton("Przewiduj")
        self.predict_button.clicked.connect(self.on_predict)
        layout.addWidget(self.predict_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_predict(self):
        try:
            data = {}
            for key, widget in self.inputs.items():
                val = widget.text().strip()
                if key in self.encoders:
                    data[key] = val  # string dla encodera
                else:
                    data[key] = float(val)  # liczba
            result = make_prediction(self.model, self.encoders, data)
            self.result_label.setText(f"Wynik: {result}")
        except Exception:
            self.result_label.setText("Błąd: sprawdź poprawność danych.")

    def on_predict(self):
        try:
            data = {}
            for key, widget in self.inputs.items():
                val = widget.text().strip()
                if not val:
                    raise ValueError(f"Pole '{key}' jest puste.")

                if key in self.encoders:
                    if val not in self.encoders[key].classes_:
                        raise ValueError(
                            f"Niepoprawna wartość dla '{key}': {val}. Dozwolone: {list(self.encoders[key].classes_)}"
                        )
                    data[key] = val
                else:
                    data[key] = float(val)

            result = make_prediction(self.model, self.encoders, data)
            self.result_label.setText(f"Wynik: {result}")
        except ValueError as ve:
            self.result_label.setText(f"Błąd: {ve}")
        except Exception as e:
            self.result_label.setText(f"Błąd ogólny: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec())
